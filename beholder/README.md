File contents: 

main_beholder.py: main script for running a MOT stream using iPhone

Disnet Utils:
- Helper functions for adapting the Twitch Streamer into MOT
    - generate_video_stream and visualize_stream were slightly edited from orignal MOT files.
        - these functions aren't great and could use a lot of work and de-bugging

Disnet:
- disnet file copied from the main directory

Transforms:
- transform file copied from the main directory 

Running a stream with inferencing:
1. Start a livestream on Streamlabs 
2. Wait for a box to appear at the top of the screen saying 30fps
3. Edit your config file as you see fit (to change to your desired output directory or weights)
    - I've been using configs/config_demo- note that the video reader objects don't apply since we instantiate a TwitchStream object
4. in `/docker` run `make main-beholder CONFIG=configs/your_config`

TODO:
1. ~~update main_beholder.py to drop all tracks that do not have a 0 or 1 label~~ 
2. ~~confirm that images are being preprocessed correctly before they are input to disnet~~
    - ~~this is in the get_dists() function in distnet_utils.py~~
3. ~~test latency of various parts of the system (particularly the get_dists() function and how long it takes to pull images from twitch)~~
4. ~~round the distance output in draw_track_boxes in output_utils.py~~

Reach Goals:
There are a couple of options we could pursue to make the codebase better
1. create a videoreader object for the TwitchStreamer like they do in the next-gen-mot repo
    - I think this would be the most challenging, but would be really nice to implement our pipeline exactly
    as they currently have it designed
2. If we don't do the above option, we could refactor the generate_video_stream function in distnet_utils.py to use ffmpeg (rn I always get broken pipe errors )
3. refactor distance to be a part of track
