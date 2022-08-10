#!/usr/bin/env python

# ---------------------------------------------------------
# (C) Copyright 2020 CACI, INC. – FEDERAL. All rights reserved.
# Direct licensing inquiries to CACI Legal Department:
#   Michael Ginsberg (Michael.ginsberg@caci.com) and
#   Stephanie Giese (sgiese@caci.com).
#
# Author(s): Zachary.Jorgensen@caci.com, Charles.Viss@caci.com
# ---------------------------------------------------------

import os
import sys
import yaml
import torch
import shutil
import logging
import warnings
import torchvision
import logging.config
from time import time, sleep
from multiprocessing import Process, Queue

# import ngmot directory
sys.path.append(os.path.join(os.getcwd(), 'ngmot'))

from beholder.beholder_utils import (save_dists,  # noqa: E402
                                     visualize_stream,  # noqa: E402
                                     get_dists,  # noqa: E402
                                     build_distnet, build_mlp)  # noqa: E402
from beholder.coordinates import get_coords_from_tracks  # noqa: E402
from beholder.kalman_distance import KalmanDistanceFilter  # noqa: E402


# setup logging. See logging.conf to change any logging preferences
fname = os.path.dirname(os.path.realpath(__file__))

with open(os.path.join(fname+'/ngmot', 'logging.conf'), 'r') as f:
    config = yaml.safe_load(f.read())
    logging.config.dictConfig(config)
logger = logging.getLogger('fmv_tracker')

warnings.filterwarnings("ignore", category=DeprecationWarning)


def main(args):
    logger.info('Loading data and models...')
    t0 = time()

    if args.use_mlflow:
        import mlflow
    # from ngmot.tracker.visualize_tracks import build_output_video
    from ngmot.tracker.tracker_utils import (construct_tracker_from_cfg,
                                             get_video)
    from beholder.output_utils import (init_output_dir,
                                       load_detections,
                                       save_tracker_output,
                                       init_output_stream,
                                       json_output_stream)

    # load detections if available
    detections_loaded = None
    if args.detection_path:
        detections_loaded = load_detections(
            args.detection_path, args.use_precomputed_features)

    video = get_video(args.video_reader_class, args.video_reader_params)
    video.start()
    stream = video

    # init output directory and settings
    init_output_dir(args, video=video)
    # build tracker
    tracker = construct_tracker_from_cfg(args)

    # initialize parallel I/O process for live demo and/or streaming output
    p, q = None, None
    if args.do_output_frames:
        p, q, displayed_frames, user_bbox, removed_detections = \
            init_output_stream(tracker, args, src_size=video.src_size,
                               interactive=args.interactive_mode)
    # init kalman distance filter
    kalman = KalmanDistanceFilter()
    model_name = 'resnet'
    if model_name == 'resnet':
        model = build_distnet()
    elif model_name == 'mlp':
        model = build_mlp()

    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )])

    # initialize parallel I/O process for streaming json output
    p2, q2 = None, None
    if args.output_json or args.output_socket:
        q2 = Queue()
        p2 = Process(target=json_output_stream,
                     args=(q2, args.output_json, args.output_dir,
                           args.output_socket, args.socket_address,
                           args.socket_port, args.socket_protocol))
        p2.start()

    t1 = time()
    logger.info('Data/model load time: {:3.2f} seconds'.format(t1 - t0))

    if args.use_mlflow:
        mlflow.log_metric('model_init_time', (t1 - t0))

    detections_all = {}
    saved_frames = []
    user_interrupt = False
    tracks = []
    dists = []
    dist_preds = []
    timestamp = time()
    logger.info("STARTING VIDEO READ")

    while True:
        if args.live_demo and displayed_frames.value == -1:
            user_interrupt = True
            break

        # load next image
        frame_num = video.current_frame
        image = stream.read()
        metadata = video.get_current_metadata()

        if image is None:
            if q is not None:
                q.put({'image': None})
            logger.info("END OF VIDEO FEED")
            break
        if os.path.exists('KILL.txt'):
            if q is not None:
                q.put({'image': None})
            os.remove('KILL.txt')
            logger.info('MOTE terminated by kill file.')
            break

        if args.interactive_mode:
            saved_frames.append(image)
            saved_frames = saved_frames[-args.output_rate:]

        saved_frames.append(image)

        # in interactive demo, ensure previous frame has been displayed before
        # processing next frame
        if args.interactive_mode:
            while (frame_num - stream.frame_start) > displayed_frames.value:
                sleep(0.001)
                if displayed_frames.value == -1:
                    user_interrupt = True
                    break

        if user_interrupt:
            break

        # check if frame processing rate should be increased or decreased
        if frame_num % tracker.frame_rate_check_interval == 0:
            elapsed_time = time() - timestamp
            timestamp = time()
            fps = tracker.frame_rate_check_interval / elapsed_time
            # logger.info('Current fps: {:.2f}'.format(fps))
            if tracker.target_frame_rate is not None:
                tracker.adjust_process_rate(fps)

        do_process_frame = (
            frame_num - stream.frame_start) % tracker.frame_process_rate == 0
        # check for user input
        if args.interactive_mode:
            # check if user added a new detection
            if not all([user_bbox[i] == 0 for i in range(4)]):
                logger.info(
                    'User input received at frame {}'.format(frame_num))
                do_process_frame = True
                tracker.add_user_detection(
                    saved_frames[-args.output_rate], frame_num-1, user_bbox[:])
                user_bbox[:] = [0, 0, 0, 0]
            # check if user deleted any detections
            while not removed_detections.empty():
                ind = removed_detections.get() - 1
                tracker.remove_user_detection(ind)

        # process frame
        detections_frame = None
        if do_process_frame:
            # load precomputed detections if available
            if detections_loaded is not None:
                detections_frame = detections_loaded.get(frame_num, [])

            # pass frame through tracker
            compute_features = (frame_num - stream.frame_start) \
                % tracker.compute_features_rate == 0
            logger.info('Processing frame (frame {})...'.format(frame_num))
            tracker_t = time()
            tracks, detections_frame = \
                tracker.process_frame(frame_num,
                                      detections_frame=detections_frame,
                                      image=image,
                                      compute_features=compute_features,
                                      metadata=metadata)
            tracker_te = time()
            logger.info('Tracker time {}'.format(tracker_te-tracker_t))
            # run new track through distance regressor
            tracker_t = time()
            with torch.no_grad():
                # track_id, label, roi, pred
                dist_preds = get_dists(
                    model, image, tracks, transforms, model_name=model_name)
            # run predictions through kalman filter
            dist_preds = kalman.update_filters(dist_preds)
            tracker_te = time()
            logger.info('Distance time {}'.format(tracker_te-tracker_t))

            # save detections for output
            if detections_frame is not None:
                detections_frame = [
                    d for d in detections_frame if d['score'] >= args.sigma_l]
                detections_all[frame_num] = detections_frame

        elif (frame_num - stream.frame_start) \
                % tracker.track_predict_rate == 0:
            # propagate tracks
            logger.info('Propagating tracks (frame {})...'.format(frame_num))
            time_t = time()
            tracks = tracker.propagate_tracks(
                nsteps=1, image=image, frame_num=frame_num, metadata=metadata)
            time_te = time()
            logger.info('Propagate time {}'.format(time_te-time_t))

        # append distance predictions to output
        dists.append(dist_preds)

        # get x, y, z offset from camera
        tracks = get_coords_from_tracks(tracks, dist_preds, video.src_size)

        if q is not None:
            if not args.draw_detections:
                detections_frame = None
            # add frame/tracks/dets to multiprocessing queue to be processed by
            # output_stream
            # q.put({'image': image, 'frame_num': frame_num,
            #       'tracks': tracks, 'detections': detections_frame})

        if args.output_json or args.output_socket:
            out_data = {'frame_num': frame_num,
                        'tracks': tracks}
            q2.put(out_data)

    # end while
    t2 = time()
    if user_interrupt:
        if args.output_frame_dir is not None and not args.output_frames:
            # delete frames intended for visualization
            shutil.rmtree(args.output_frame_dir)
        return

    final_tracks = tracker.finish_tracks(include_features=args.output_features)
    if args.use_mlflow:
        mlflow.log_metric('tracks_finished', len(tracker.tracks_finished))

    logger.info("Finished. Framerate: " +
                str(int((frame_num - stream.frame_start) / (t2 - t1))) +
                " fps!")
    if args.use_mlflow:
        mlflow.log_metric('tracker_fps', float(
            frame_num - stream.frame_start) / (t2 - t1))

    # process and save tracker output as specified by config
    save_tracker_output(args, final_tracks,
                        final_frame_num=frame_num, detections=detections_all)
    save_dists(args, dists)
    if p is not None:
        # wait for processed frames to finish being saved / displayed
        p.join()
        q.close()
    if args.output_socket:
        q2.put(-1)
        p2.join()
        q2.close()

    # build output video
    if args.output_video:
        visualize_stream(saved_frames,
                         stream,
                         tracks=final_tracks,
                         dists=dists,
                         output_file=os.path.join(
                             args.output_dir, 'output_video.mp4'),
                         dets=None,
                         tracks_src_size=None,
                         draw_track_history=args.draw_track_history,
                         codec=args.ffmpeg_codec,
                         attributes_max=args.attributes_max,
                         frame_rate=args.output_frame_rate,
                         attributes_thresh=args.attributes_thresh)

    video.close()
    tracker.close()
    logger.info('END OF TRACKER')
    return 0


if __name__ == '__main__':
    import argparse
    from ngmot.parse_cfg import parse_cfg

    parser = argparse.ArgumentParser()
    parser.add_argument('cfg', help='path to config file containing arguments')
    cmd_args = parser.parse_args()

    # load args from cfg file
    args = parse_cfg(cmd_args.cfg)

    if args.use_mlflow:
        import mlflow
        logger.info('Tracking in MlFlow...')
        mlflow.set_experiment('NGMOT_tracker')
        mlflow.log_artifact(cmd_args.cfg)

    if args.profile_tracker:
        import cProfile
        import pstats
        from ngmot.tracker.visualize_profile import visualize_profile

        def profile_tracker(function_call):
            temp = os.path.join(args.output_dir, 'profile')
            cProfile.run(function_call, temp)

            # re-save cProfile output in a human-readable format
            profile_fn = os.path.join(args.output_dir, 'profile.txt')
            with open(profile_fn, 'w') as f:
                p = pstats.Stats(temp, stream=f)
                p.strip_dirs().sort_stats('cumtime').print_stats()
            os.remove(temp)

            # build and save charts from profile output
            profile_charts = os.path.join(args.output_dir, 'profile.png')
            visualize_profile(profile_fn, profile_charts)
            if args.use_mlflow:
                mlflow.log_artifact(profile_fn)
                mlflow.log_artifact(profile_charts)

    # call main function
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if args.profile_tracker:
            profile_tracker('main(args)')
        else:
            main(args)

    if args.use_mlflow:
        mlflow.log_artifact('tracker.log')
