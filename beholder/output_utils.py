# (C) Copyright 2020 CACI, INC. – FEDERAL. All rights reserved.
# Direct licensing inquiries to CACI Legal Department:
#   Michael Ginsberg (Michael.ginsberg@caci.com) and
#   Stephanie Giese (sgiese@caci.com).
#
# Author(s) Zachary.Jorgensen@caci.com, Charles.Viss@caci.com

"""
I/O-related functionality used by MOTE's driver script
"""

import os
import shutil
import numpy as np
import pandas as pd
import csv
import torch
import logging
import socket
from PIL import Image, ImageDraw, ImageFont
import subprocess
import time
import json
import pickle
import requests
from multiprocessing import Process, Queue, Value, Array

from ngmot.tracker.tracker_utils import scale_bbox

logger = logging.getLogger('fmv_tracker.util')


COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255),
          (192, 0, 192), (192, 192, 0), (0, 192, 192),
          (128, 0, 0), (0, 128, 0), (0, 0, 128),
          (128, 0, 128), (128, 128, 0), (0, 128, 128)]
COLOR_DICT = {'red': (255, 0, 0), 'green': (0, 255, 0), 'blue': (0, 0, 255),
              'gray': (96, 96, 96), 'black': (0, 0, 0),
              'white': (255, 255, 255), 'brown': (165, 42, 42),
              'yellow': (255, 255, 0), 'orange': (255, 128, 0),
              'silver': (196, 196, 196)}


def get_dist(track_id, dists):
    for d in dists:
        if d[0] == track_id:
            return d[3]
    return 0


def draw_track_boxes(image, tracks, dists=None, dets=None, src_size=None,
                     attributes_max=0, attributes_thresh=0.5,
                     clicked_track_id=-1):
    """
        Draws current bounding box for each track active in the current frame.
        Labels the bounding boxes with their corresponding track IDs.
        Args:
            image: PIL image for the current frame
            tracks: list of dicts representing the current box for each track
            dets: detections for the current frame
            src_size: (w,h)-size of source imagery (needed only if image has
                been resized)
            attributes_max: maximum number of attributes to be displayed per
                track
            attributes_thresh: score threshold for attribute to be displayed
            clicked_track_id: id of any user-selected track
        Returns:
            frame: PIL image with the boxes and IDs drawn on top.
    """
    frame = image.copy()
    draw = ImageDraw.Draw(frame)
    font = ImageFont.truetype('beholder/Lato-Heavy.ttf', 24)
    if dists is None:
        dists = []
        for row in tracks:
            dists.append([0 for _ in range(len(row))])

    for row in tracks:
        if row['label_name'] is not None and row['label_name'] == 'HIDDEN':
            continue
        color = COLORS[int(row['track_id']) % len(COLORS)]
        bbox = [int(row['roi'][0]), int(row['roi'][1]),
                int(row['roi'][2]), int(row['roi'][3])]
        if src_size is not None:
            bbox = scale_bbox(bbox, src_size, image.size)

        # Draw bounding box
        draw.rectangle(bbox, outline=color, fill=None, width=2)

        # Highlight clicked track id
        if clicked_track_id == row['track_id']:
            draw.rectangle(bbox, outline=(255, 255, 255), fill=None, width=4)
        dist = get_dist(row['track_id'], dists)
        # Set box text
        if row['label_name'] is not None:
            txt = '{} ({}) dist: {}'.format(
                int(row['track_id']), row['label_name'], round(dist, 2))
        elif row['label_name'] is not None:
            txt = '{} ({}) dist: na'.format(
                int(row['track_id']), row['label_name'])

        else:
            txt = '{} (class {})'.format(int(row['track_id']), row['label'])
        # Get size of ID text in pixels
        txtsz = draw.textsize(txt, font=font)

        # Draw solid rectangle for ID text background
        draw.rectangle([bbox[0], bbox[1], bbox[0] + txtsz[0] + 4,
                       bbox[1] + txtsz[1]], fill=color, outline=color)

        # Draw track ID number and current track label
        draw.text((bbox[0] + 4, bbox[1]), txt, font=font, fill=(0, 0, 0))
        draw.text((bbox[0] + 3, bbox[1] - 1), txt,
                  font=font, fill=(224, 224, 224))

        # write text attributes below the box
        if row.get('top_attributes') is not None:
            offset = 12
            for i, att in enumerate(row['top_attributes']):
                color = COLOR_DICT.get(att['value'], (255, 255, 255))
                if i >= attributes_max or att['score'] < attributes_thresh:
                    break
                att_txt = '{}: {} ({:.2f})'.format(
                    att['name'], att['value'], att['score'])
                draw.text((bbox[0] + 4, bbox[1] + offset),
                          att_txt, font=font, fill=color)
                offset += 12
                if 'bbox' in att:
                    draw.rectangle([att['bbox'][0], att['bbox'][1],
                                   att['bbox'][2], att['bbox'][3]],
                                   outline=color)

    if dets is not None:
        for det in dets:
            color = (255, 255, 255)
            # Draw bounding box
            bbox = [det['roi'][0], det['roi'][1], det['roi'][2], det['roi'][3]]
            if src_size is not None:
                bbox = scale_bbox(bbox, src_size, image.size)
            draw.rectangle(bbox, outline=color, fill=None, width=1)

    return frame


def socket_connect(addr='127.0.0.1', port=8093, protocol='UDP'):
    """
        Establishes connection to socket for output data stream
        Args:
            address: IP address
            port: connection port
            protocol: UDP or TCP
        Returns:
            socket object if successful, else None
    """
    try:
        if protocol.upper() == 'TCP':
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect((addr, port))
        elif protocol.upper() == 'UDP':
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        else:
            raise ValueError('Invalid protocol: {}'.format(protocol))
        return s
    except Exception as e:
        logger.info("[SOCKET ERROR] Could not create socket connection at {}:"
                    "{}".format(addr, port))
        logger.info(e)
        return None


def send_to_socket(data, sock, protocol='UDP', addr=None, port=None):
    """
    Send json data to socket
    Args:
        data: jsonifyable data to be sent
        sock: destination socket
    """
    try:
        data = json.dumps(data) + '\n'
        if protocol.upper() == 'TCP':
            msg = bytes(data, encoding="utf8")
            sock.sendall(msg)
            return True
        if protocol.upper() == 'UDP':
            if addr is None or port is None:
                raise ValueError(
                    'Address and port must be provided to send over UDP')
            sync_bytes = b'\x24\x40'
            msg_data = bytes(data, encoding='utf8')
            msg_length = len(msg_data).to_bytes(2, byteorder='big')
            msg = sync_bytes + msg_length + msg_data
            sock.sendto(msg, (addr, port))
            return True
        raise ValueError('Invalid protocol: {}'.format(protocol))
    except Exception as e:
        logger.info("[SOCKET ERROR] Could not send over socket.")
        logger.info(e)
        return False


def close_socket(sock):
    """
    Close connection to the provided socket
    """
    try:
        if sock is not None:
            sock.shutdown(socket.SHUT_RDWR)
            sock.close()
    except Exception as e:
        logger.info(e)


def append_to_json(data, json_fn):
    """
    Write data to a new or exiting json file
    Args:
        data: json data to be written
        json_fn: output json file name
    """
    with open(json_fn, 'ab+') as f:
        f.seek(0, 2)  # Go to the end of file
        if f.tell() == 0:  # Check if file is empty
            # If empty, write an array
            f.write(bytes(json.dumps([data]), encoding='utf8'))
        else:
            f.seek(-1, 2)
            f.truncate()  # Remove the last character, open the array
            f.write(bytes(' , ', encoding='utf8'))  # Write the separator
            # Dump the dictionary
            f.write(bytes(json.dumps(data), encoding='utf8'))
            f.write(bytes(']', encoding='utf8'))  # Close the array


def json_output_stream(q, output_json=False, output_dir='outputs',
                       output_socket=False, addr='127.0.0.1', port=8093,
                       protocol='UDP'):
    """
        Parallel process for streaming json output to network address or
        output file
        Args:
            q: Multiprocessing Queue of track json data to be streamed
            output_json: boolean indicating whether to stream output to json
                file
            output_dir: directory to place json output file
            output_socket: boolean indicating whether to stream output to
                network address
            addr: IP address for network output
            port: port for network output
            protocol: TCP or UDP; protocol for network output
    """
    if output_socket:
        sock = socket_connect(addr, port, protocol)
    if output_json and os.path.exists(
            os.path.join(output_dir, 'live_tracks.json')):
        os.remove(os.path.join(output_dir, 'live_tracks.json'))
    while True:
        if q.empty():
            # wait for track data to become available
            time.sleep(0.001)
        else:
            data = q.get()
            if data == -1:
                if output_socket:
                    close_socket(sock)
                break
            if output_json:
                append_to_json(data, os.path.join(
                    output_dir, 'live_tracks.json'))
            if output_socket:
                if sock is None:
                    sock = socket_connect(addr, port)
                send_to_socket(data, sock, protocol, addr, port)


def get_ffmpeg_handle(ffmpeg_destination, frame_size, frame_rate=30,
                      vcodec='libx264', bitrate='6000k',
                      bufsize=10**8, ffmpeg_bin='/usr/bin/ffmpeg'):
    """
    Build, start, and return an FFMPEG process for writing a video using the
        provided parameters.
    Args:
        ffmpeg_destination: name of output video file or stream
        frame_size: (w, h) dimention of video frames
        frame_rate: framerate of video in fps
        vcodec: code for output video. options: mpeg4, mp4v, libx264,
            mpeg2video, mpeg1video, msmpeg4v2
        bitrate: video bitrate of the output file in bit/s
        bufsize: buffer size for ffmpeg process handle
        ffmpeg_bin: string command to launch ffmpeg
    """
    if frame_rate is None:
        raise ValueError('Must provide output frame rate for ffmpeg output.')
    ffmpeg_command = [ffmpeg_bin, '-y',
                      # Input
                      '-f', 'rawvideo', '-vcodec', 'rawvideo',
                      '-s', str(int(frame_size[0])) + \
                      'x'+str(int(frame_size[1])),
                      '-pix_fmt', 'bgr24', '-r', str(float(frame_rate)),
                      '-i', '-',
                      # Outputs
                      '-vcodec', vcodec,
                      '-r', str(float(frame_rate)),
                      '-b:v', bitrate,
                      '-pix_fmt', 'yuv420p',
                      '-crf', '24']
    if ffmpeg_destination.endswith('.ts') or \
            ffmpeg_destination.startswith('udp'):
        ffmpeg_command.extend(['-f', 'mpegts'])
    ffmpeg_command.append(ffmpeg_destination)
    ffmpeg_handle = subprocess.Popen(ffmpeg_command, stdin=subprocess.PIPE,
                                     stderr=subprocess.STDOUT,
                                     stdout=subprocess.DEVNULL,
                                     bufsize=bufsize)
    return ffmpeg_handle


def write_frame_to_ffmpeg(ffmpeg_handle, frame):
    """
    Write video frame to ffmpeg output process
    Args:
        ffmpeg_handle: active ffmpeg process
        frame: PIL image
    """
    try:
        ffmpeg_handle.stdin.write(np.array(frame)[:, :, ::-1].tobytes())
        ffmpeg_handle.stdin.flush()
    except Exception as e:
        logger.error("\n\nEXCEPTION:")
        logger.error(e)
        logger.error("Error writing frame to ffmpeg: " +
                     str(ffmpeg_handle.returncode))


def close_ffmpeg_handle(ffmpeg_handle):
    """
    Close active ffmpeg process
    Args:
        ffmpeg_handle: active ffmpeg process
    """
    ffmpeg_handle.stdin.close()
    ffmpeg_handle.wait()
#   # Try to gracefully kill it, then force kill it. Seems to be an issue with
#     h264 codec
#     try:
#         subprocess.Popen(["/bin/kill", str(ffmpeg_handle.pid)])
#         time.sleep(1.0)
#         subprocess.Popen(["/bin/kill", "-9", str(ffmpeg_handle.pid)])
#     except IOError:
#         traceback.print_exc()


def send_frame_to_flask_url(cv2_img, flask_url):
    """
    Args:
        cv2_image: cv2 image or numpy array of frame
        flask_url: url for flask server
    """
    try:
        img_data = pickle.dumps(cv2_img)
        requests.post(flask_url, data=img_data)
    except Exception as e:
        logger.error("\n\nError sending to flask server:")
        logger.error(e)


class InteractiveDisplay():
    """
    Wrapper class for processing user input received by CV2 display.
    """

    def __init__(self, src_size=None, user_bbox=None, removed_detections=None,
                 initial_query=None):
        """
        Args:
            user_bbox: shared multiprocessing Array for when user provides
                input in interactive mode
            removed_detections: multiprocessing Queue for when user removes one
                of their selections
            inital_query: list of any initial query image filenames provided by
                the user
        """

        if initial_query is None:
            initial_query = []

        self.user_bbox = user_bbox
        self.removed_detections = removed_detections
        self.src_size = src_size
        self.n_selected = 0
        self.active_samples = set([])
        for query in initial_query:
            self.add_query(query)

    def add_query(self, query):
        """
        Displays query selection and adds to active set
        Args:
            query: cv2 image of query or str path to a query image
        """
        import cv2
        if isinstance(query, str):
            query = Image.open(query)
            query = cv2.cvtColor(np.array(query), cv2.COLOR_RGB2BGR)
        self.n_selected += 1
        self.active_samples.add(self.n_selected)
        cv2.imshow('Sample {}'.format(self.n_selected), query)

    def get_input(self, key, frame, cv2_img):
        """
        Process any additional user input from cv2 display
        (bbox selection on main display or closing of any existing queries)
        Args:
            key: key pressed by user
            frame: original PIL Image of frame before track boxes were drawn
            cv2_image: cv2 image with tracks drawn that has been displayed to
            user
        """
        import cv2
        if key == ord('s'):
            bbox = cv2.selectROI(
                'MOTE', cv2_img, fromCenter=False, showCrosshair=True)
            bbox = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
            display_size = (cv2_img.shape[1], cv2_img.shape[0])

            # if the user gives no input, continue as normal. otherwise,
            # process the user input
            if not all(x == 0 for x in bbox):
                # display selected sample
                frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
                frame = cv2.resize(frame, display_size)
                query = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                self.add_query(query)

                # convert back to original coordinates
                if self.src_size is not None:
                    bbox = scale_bbox(bbox, display_size, self.src_size)

                # return to main program via shared array
                self.user_bbox[:] = bbox[:]

        # check if any detections have been removed by user
        removed = []
        for i in self.active_samples:
            if cv2.getWindowProperty(
                    'Sample {}'.format(i), cv2.WND_PROP_VISIBLE) < 1:
                logger.info('Removing detection {}!'.format(i))
                self.removed_detections.put(i)
                removed.append(i)
        for i in removed:
            self.active_samples.remove(i)


def output_frame(frame, frame_num, detections, tracks,
                 displayed_frames=None, src_size=None,
                 display_size=(1280, 720), display_rate=1,
                 draw_detections=False, attributes_max=0,
                 attributes_thresh=0.5, frame_dir=None, cv2_display=False,
                 interactive_display=None, flask_url=None, ffmpeg_handle=None,
                 print_tracks=False):
    """
    Draw tracks on frame and output to specified destinations.
    See output_stream below for more parameter details.
    Args:
        frame: raw PIL image of video frame
        frame_num: video frame number
        detections: list of raw detections
        tracks: current tracks for frame
        interactive_display: InteractiveDisplay object if using interactive
            viewer else None
        ffmpeg_handle: FFMPEG process object if using an ffmpeg destination
            output else None
    """
    if frame_dir:
        frame.save(os.path.join(frame_dir, '{:06d}.jpg'.format(frame_num)))
    if print_tracks:
        print(tracks)

    if frame_num % display_rate == 0:
        dets_to_draw = detections if draw_detections else None
        drawn = draw_track_boxes(
            frame, tracks, dets_to_draw, src_size, attributes_max,
            attributes_thresh)
        if display_size is not None:
            drawn = drawn.resize(display_size)
        if cv2_display or flask_url:
            import cv2
            cv2_img = cv2.cvtColor(np.array(drawn), cv2.COLOR_RGB2BGR)
        if ffmpeg_handle:
            write_frame_to_ffmpeg(ffmpeg_handle, drawn)
        if flask_url:
            send_frame_to_flask_url(cv2_img, flask_url)
        if cv2_display or interactive_display:
            has_display = os.environ.get('DISPLAY', None) is not None
            if not has_display:
                logger.info('WARNING: No display available for CV2')
            else:
                cv2.imshow('MOTE', cv2_img)
                key = cv2.waitKey(1) & 0xFF

                # check whether user pressed ESC or q to quit, or closed the
                # window
                if key == 27 or key == ord('q') or cv2.getWindowProperty(
                        'MOTE', cv2.WND_PROP_VISIBLE) < 1:
                    cv2.destroyAllWindows()
                    if displayed_frames is not None:
                        displayed_frames.value = -1
                elif interactive_display is not None:
                    interactive_display.get_input(key, frame, cv2_img)

    if displayed_frames is not None and displayed_frames.value >= 0:
        displayed_frames.value += 1


def output_stream(q, frame_rate=None, src_size=None, display_size=(1200, 720),
                  display_rate=2, displayed_frames=None,
                  interactive_mode=False, interactive_params=None,
                  draw_detections=False, attributes_max=0,
                  attributes_thresh=0.5, cv2_display=False,
                  ffmpeg_destination=None, ffmpeg_codec='libx264',
                  flask_url=None, frame_dir=None, print_tracks=False):
    """
    Function to be launched as a parallel process which draws boxes on
    processed frames and displays the frames in a cv2 window and/or sends them
    to the specified destination at the appropriate frame rate.
    Args:
        q: multiprocessing Queue where each entry is a dictionary of the form:
            {'image': PIL image of frame, 'frame_num': video frame number,
             'tracks': current tracks, 'detections': raw frame detections
                (optional)}
        frame_rate (int): maximum frame rate the processed images will be
            displayed (if frame_rate is None, imgaes will be displayed as
            processed)
        src_size: (w,h)-size of source imagery (needed only if image has been
            resized)
        display_size: (w,h)-size of output
        display_rate (int): rate at which frames are displayed
        displayed_frames: a multiprocessing Value indicating how many frames
            have been displayed
        interactive_mode (boolean): whether or not user can provide input via
            cv2 window
        interactive_params (dict): params for interactive display
        draw_detections: indicate whether to draw raw detections on output
        attributes_max: maximum number of attributes to be displayed
        attributes_thresh: min score thresh of attributes to be displayed
        cv2_display: boolean indicating whether to have live cv2 display
        ffmpeg_destination: ffmpeg-style stream or video destination
        ffmpeg_codec: codec for ffmpeg
        flask_url: url for actively running flask server destination
        frame_dir: if provided, directory for saving frames to disk
        print_tracks (bool): print tracks to stdout
    """

    if interactive_params is None:
        interactive_params = {}

    s_per_frame = display_rate / frame_rate if frame_rate is not None else None
    interactive_display = None
    ffmpeg_handle = None

    if frame_dir:
        os.makedirs(frame_dir, exist_ok=True)
    if interactive_mode:
        interactive_display = InteractiveDisplay(
            src_size=src_size, **interactive_params)
    if ffmpeg_destination:
        ffmpeg_handle = get_ffmpeg_handle(ffmpeg_destination,
                                          frame_size=display_size,
                                          frame_rate=int(
                                            frame_rate / display_rate),
                                          vcodec=ffmpeg_codec)

    # loop until all frames have been processed
    t0 = time.time()
    while True:
        if q.empty():
            time.sleep(0.001)  # wait for a processed frame to be available
        else:
            d = q.get()
            frame, frame_num, detections, tracks = d['image'], d.get(
                'frame_num'), d.get('detections'), d.get('tracks')

            if frame is None:
                break  # End of stream

            if cv2_display and s_per_frame is not None and \
                    frame_num % display_rate == 0:
                elapsed_time = time.time() - t0
                t0 = time.time()
                if elapsed_time < s_per_frame:
                    # don't display frames faster than the specified framerate
                    time.sleep(s_per_frame - elapsed_time)

            output_frame(frame, frame_num, detections, tracks,
                         displayed_frames, src_size, display_size,
                         display_rate, draw_detections, attributes_max,
                         attributes_thresh, frame_dir, cv2_display,
                         interactive_display, flask_url,
                         ffmpeg_handle, print_tracks)
            if displayed_frames is not None and displayed_frames.value == -1:
                break

    if cv2_display:
        import cv2
        cv2.destroyAllWindows()
        cv2.waitKey(1)
    if ffmpeg_destination:
        close_ffmpeg_handle(ffmpeg_handle)
    logger.info('END OF OUTPUT STREAM')
    return 0


def init_output_stream(tracker, args, src_size=None, interactive=False,
                       queue_size=2000):
    """
    Initialize and start an output stream process using provided parameters.
    Args:
        tracker: CACITracker object
        args: tracker config from parse_cfg
        src_size: size of source video (w, h)
        interactive: boolean indicating whether to use an interactive display
        queue_size: maximum size of Multiprocessing Queue to be used to send
            frames to process
    Returns:
        p: Multiprocessing Process
        q: Multiprocessing Queue
        displayed_frames: shared integer value indicating number of frames
            displayed or whether to terminate process if value is -1.
        user_bbox: multiprocessing array for receiving user bbox input
        removed_detections: multiprocessing queue containing user detections to
            be removed
    """
    # multiprocessing queue for frames to be processed
    q = Queue(queue_size)
    # multiprocessing integer Value indicating the number of frames that have
    # been processed
    displayed_frames = Value('i', 0)

    user_bbox, removed_detections = None, None
    if interactive:
        # multiprocessing array for receiving user bbox input
        user_bbox = Array('i', [0, 0, 0, 0])
        # multiprocessing queue for which user-defined detections have been
        # deleted
        removed_detections = Queue()

    p = Process(target=output_stream,
                args=(q, args.output_frame_rate, src_size,
                      args.output_size, args.output_rate,
                      displayed_frames, interactive,
                      {'user_bbox': user_bbox,
                       'removed_detections': removed_detections,
                       'initial_query': args.initial_query},
                      args.draw_detections, args.attributes_max,
                      args.attributes_thresh, args.live_demo,
                      args.ffmpeg_destination, args.ffmpeg_codec,
                      args.flask_url, args.output_frame_dir,
                      args.print_tracks))
    p.start()
    return p, q, displayed_frames, user_bbox, removed_detections


def init_output_dir(args, video=None):
    """
    Set up output directory and configure settings based on provided config and
    video
    Args:
        args: tracker config from parse_cfg
        video: VideoReader (optional)
    """
    # set up output directory
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if args.detection_path is None and os.path.exists(
            os.path.join(args.output_dir, 'det.txt')):
        os.remove(os.path.join(args.output_dir, 'det.txt'))
    if not os.path.exists(os.path.join(args.output_dir, 'cfg.yml')):
        shutil.copyfile(args.config_fn, os.path.join(
            args.output_dir, 'cfg.yml'))
    elif not os.path.samefile(args.config_fn,
                              os.path.join(args.output_dir, 'cfg.yml')):
        shutil.copyfile(args.config_fn, os.path.join(
            args.output_dir, 'cfg.yml'))

    # update output settings based on provided video
    if video is not None:
        if args.output_size is None:
            args.output_size = video.src_size
        if args.display_size is None:
            args.display_size = video.src_size
        if args.output_frame_rate is None:
            args.output_frame_rate = video.fps
        if (args.output_video or args.output_thumbnails) and video._is_stream:
            # temporarily save frames to disk to support post-processing
            # visualization
            args.output_frame_dir = os.path.join(args.output_dir, 'frames')
            if os.path.exists(args.output_frame_dir):
                shutil.rmtree(args.output_frame_dir)


def save_detections(detections, fn, include_features=False):
    """
    Saves detections as a numpy array.
    Each row of the numpy array will have the following form:
        [frame, label, x1, y1, x2, y2, score, features].
    An exception will be reaised if include_features is True but features are
        not found in the detections.
    Args:
        detections (dict): dictionary of the form {frame_num: [list of
            detections in frame]}
        fn (str): save path for detections (*.npy)
        include_features (boolean): indicate whether detection features should
            be saved as well
    """
    dets_arr = []
    for frame_num in detections:
        for d in detections[frame_num]:
            row = [d['frame'], d['label'], d['roi'][0], d['roi'][1],
                   d['roi'][2], d['roi'][3], d['score']]
            if include_features:
                try:
                    row.extend(d['features'].tolist())
                except AttributeError as error:
                    raise RuntimeError(
                        'Detection features unavailable for output.') \
                        from error
            dets_arr.append(row)
    dets = np.array(dets_arr)
    np.save(fn, dets)


def load_detections(detections_fn, include_features=False):
    """
    Loads detections stored in a numpy array.
    Assumes each row of the numpy array has the following form:
        [frame, label, x1, y1, x2, y2, score, features].
    An exception will be reaised if include_features is True but features are
        not found in the numpy array.
    Args:
        detections_fn (str): path to detections file (*.npy)
        include_features (boolean): indicate whether detection features should
            be returned as well
    Returns:
        detections: dictionary of the form {frame_num: [list of detections in
            frame]}
    """
    # for backwards compatibility, first check if the detections file is a .csv
    if detections_fn.endswith('.txt') or detections_fn.endswith('.csv'):
        if include_features:
            raise RuntimeError(
                'Features not available in {}'.format(detections_fn))
        return load_mot(detections_fn)

    detections = {}
    arr = np.load(detections_fn)
    if include_features and arr.shape[1] < 8:
        raise RuntimeError(
            'Features not found in detections file {}.'.format(detections_fn))

    for row in arr:
        frame_num = int(row[0])
        det = {'frame': frame_num,
               'label': int(row[1]),
               'roi': [int(row[2]), int(row[3]), int(row[4]), int(row[5])],
               'score': row[6],
               'centroid': [(row[2] + row[4]) / 2, (row[3] + row[5]) / 2],
               }
        if include_features:
            det['features'] = row[7:]

        if frame_num in detections:
            detections[frame_num].append(det)
        else:
            detections[frame_num] = [det]

    return detections


def load_mot(detections):
    """
    Loads detections stored in a mot-challenge like formatted CSV or numpy
    array
        (fieldNames = ['frame', 'id', 'x', 'y', 'w', 'h', 'score']).
    Args:
        detections
    Returns:
        list: list containing the detections for each frame.
    """
    data = {}
    if isinstance(detections, str):
        arr = np.genfromtxt(detections, delimiter=',', dtype=np.float32)
    else:
        # assume it is an array
        assert isinstance(
            detections, np.ndarray), "only numpy arrays or *.csv paths are " \
            "supported as detections."
        arr = detections.astype(np.float32)
    for row in arr:
        frame_num = int(row[0])
        det = {'frame': frame_num,
               # this label is always -1 in the case of MOT style detections
               'label': int(row[1]),
               'roi': [int(row[2]), int(row[3]), int(row[2] + row[4]),
                       int(row[3] + row[5])],
               'centroid': [row[2] + (row[4] / 2), row[3] + (row[5] / 2)],
               'score': row[6],
               }
        if frame_num in data:
            data[frame_num].append(det)
        else:
            data[frame_num] = [det]
    return data


def save_to_csv(tracks, out_path, labels_path, include_tracklet_id=True,
                include_header=True):
    """
    Saves tracks and track labels to CSV files.
    Class labels are stored separate from the general tracks output to prevent
    conflicts in the MOT metrics computations.
    Args:
        tracks (list): list of tracks to store.
        out_path (str): path to output csv file.
        labels_path (str): path to labels csv file.
    """

    with open(out_path, "w") as ofile:
        if include_tracklet_id:
            field_names = ['frame', 'tracklet_id', 'track_id',
                           'x', 'y', 'w', 'h', 'score', 'wx', 'wy', 'wz']
        else:
            field_names = ['frame', 'track_id', 'x',
                           'y', 'w', 'h', 'score', 'wx', 'wy', 'wz']
        odict = csv.DictWriter(ofile, field_names)
        if include_header:
            odict.writeheader()
        for track in tracks:
            for bbox in track['detections']:
                row = {'track_id': track['track_id'],
                       'frame': bbox['frame_num'],
                       'x': bbox['roi'][0],
                       'y': bbox['roi'][1],
                       'w': bbox['roi'][2] - bbox['roi'][0],
                       'h': bbox['roi'][3] - bbox['roi'][1],
                       'score': 1.0,  # bbox['score'],
                       'wx': -1,
                       'wy': -1,
                       'wz': -1}
                if include_tracklet_id:
                    row['tracklet_id'] = track['tracklet_id']
                odict.writerow(row)

    with open(labels_path, 'w') as ofile:
        field_names = ['tracklet_id', 'track_id', 'label', 'label_name']
        odict = csv.DictWriter(ofile, field_names)
        odict.writeheader()
        for track in tracks:
            label = track['label']
            label_name = track['label_name']
            row = {'tracklet_id': track['tracklet_id'],
                   'track_id': track['track_id'], 'label': label}
            if label_name is not None:
                row['label_name'] = label_name
            odict.writerow(row)


def save_all_tracks_info(tracks, fn):
    """
    Save all finished track info (including detections and features) as lists
    of dictionaries.
    Each track is a list of dictionaries of the form:
        [{'roi': bbox, 'frame': frame_num, 'track_id': track_id,
          'label': track_label, 'label_name': track_label_name
          'features': det features or None, 'attributes': det attributes or
          None}
         for f in frame_range]
    """
    torch.save(tracks, fn)


def save_tracks_df(tracks, fn):
    """
    Save all finished track info (including detections and features) as a
    pandas dataframe.
    """
    df = build_tracks_df(tracks)
    df.to_pickle(fn)


def save_tracks_json(tracks, fn):
    """
    Save all finished track info (including detections and features) as a json
    file.
    """
    with open(fn, 'w') as f:
        json.dump(tracks, f)


def build_tracks_df(tracks):
    """
    Build pandas dataframe from list of finished tracks
    """
    # build pandas dataframe to save tracks info
    data = {'track_num': [], 'tracklet_id': [], 'track_id': [],
            'frame_num': [], 'roi': [], 'features': [], 'label': [],
            'label_name': [], 'attributes': [], 'top_attributes': []}
    for i, track in enumerate(tracks):
        for det in track['detections']:
            data['track_num'].append(i)
            data['tracklet_id'].append(track['tracklet_id'])
            data['track_id'].append(track['track_id'])
            data['frame_num'].append(det['frame_num'])
            data['roi'].append(det['roi'])
            data['features'].append(det['features'])
            data['label'].append(track['label'])
            data['label_name'].append(track['label_name'])
            data['attributes'].append(det['attributes'])
            data['top_attributes'].append(det['top_attributes'])
    df = pd.DataFrame(data, columns=data.keys())
    return df


def save_tracker_output(args, final_tracks, final_frame_num=None,
                        detections=None):
    """
    Save tracker output in various formats as specified by tracker args from
    parse_cfg
    Args:
        args: tracker config from parse_cfg
        final_tracks (list): postprocessed tracks from CACITracker
        final_frame_num (int): last frame num processed by tracker
        detections (dict): all raw detections from tracker
    """
    if args.use_mlflow:
        import mlflow

    # save track output and labels
    output_fn = os.path.join(args.output_dir, 'tracks_output.csv')
    labels_fn = os.path.join(args.output_dir, 'track_labels.csv')
    save_to_csv(final_tracks, output_fn, labels_fn,
                args.output_tracklet_ids, args.output_csv_header)
    if args.use_mlflow:
        mlflow.log_artifact(output_fn)

    # log final frame processed
    if final_frame_num is not None:
        with open(os.path.join(args.output_dir, 'last_frame.txt'), 'w') as f:
            f.write(str(final_frame_num))

    # save detection/features output
    if args.output_detections and detections is not None:
        dets_file = os.path.join(args.output_dir, 'detections.npy')
        save_detections(detections, dets_file,
                        include_features=args.output_features)

    # save all finished tracks info (including dets and features) for archival
    # purposes
    if args.output_json:
        json_fn = os.path.join(args.output_dir, 'final_tracks.json')
        save_tracks_json(final_tracks, json_fn)
    if args.output_all_tracks_info:
        tracks_fn = os.path.join(args.output_dir, 'track_info.pth')
        save_all_tracks_info(final_tracks, tracks_fn)

    # save all finished tracks info (including dets and features) for archival
    # purposes in dataframe
    if args.output_tracks_dataframe:
        df_fn = os.path.join(args.output_dir, 'tracks_df.pkl')
        save_tracks_df(final_tracks, df_fn)

    # compute metrics
    if args.compute_metrics:
        logger.info("Computing metrics...")
        metrics_fn = os.path.join(args.output_dir, 'metrics_df.pkl')
        from metrics import compute_mot_metrics
        metrics_summary, _ = compute_mot_metrics(output_fn,
                                                 args.metrics_ground_truth,
                                                 metrics_fn)
        if args.use_mlflow:
            mlflow.log_artifact(metrics_fn)
            mlflow.log_metrics(metrics_summary.loc['OVERALL'])
