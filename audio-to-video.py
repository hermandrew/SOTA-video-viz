import math
import cairo
import numpy as np
import librosa
import os, shutil
import urllib.request
import progressbar
import moviepy.editor as editor
import argparse

SCRATCH_DIR = './scratch'
OUTPUT_DIR = './out'
DOWNLOAD_DIR = './dl'
VERBOSE = True
CLEANUP = True

FRAMERATE = 30

AUDIO_LEFT_CHANNEL = None
AUDIO_RIGHT_CHANNEL = None

WIDTH, HEIGHT = 600, 600
L_RADIUS, R_RADIUS, C_RADIUS = .4, .2, .3

def make_frame(t):
    if VERBOSE:
        print("STARTING TO DRAW FRAMES")

    bar = progressbar.ProgressBar(maxval=AUDIO_LEFT_CHANNEL.size, \
                                  widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])

    if VERBOSE:
        bar.start()

    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, WIDTH, HEIGHT)
    ctx = cairo.Context(surface)
    ctx.scale(WIDTH, HEIGHT)  # Normalizing the canvas
    ctx.rectangle(0, 0, 1, 1)
    ctx.set_source_rgb(0, 0, 0)
    ctx.fill()
    ctx.translate(0.5, 0.5)  # Changing the current transformation matrix

    draw_radial_waveform(AUDIO_LEFT_CHANNEL[t], ctx, radius=(L_RADIUS))
    draw_radial_waveform(AUDIO_RIGHT_CHANNEL[t], ctx, radius=(R_RADIUS), radial_offset=math.pi/4)
    ctx.arc(0, 0, C_RADIUS, 0, 2 * math.pi)
    ctx.set_source_rgb(1, 1, 1)
    ctx.stroke()

    buf = surface.get_data()
    data = np.ndarray(shape=(WIDTH, HEIGHT),
                         dtype=np.uint32,
                         buffer=buf)

    if VERBOSE:
        bar.finish()
        print('DRAWING FRAMES FINISHED')

    return data

def draw_stereo_output(left, right):
    if VERBOSE:
        print("STARTING TO DRAW FRAMES")

    WIDTH, HEIGHT = 600, 600
    L_RADIUS, R_RADIUS, C_RADIUS = .4, .2, .3

    reset_dir(SCRATCH_DIR)

    paths = []

    bar = progressbar.ProgressBar(maxval=left.size, \
                                  widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])

    if VERBOSE:
        bar.start()

    for fftIndex, l in enumerate(left):
        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, WIDTH, HEIGHT)
        ctx = cairo.Context(surface)
        ctx.scale(WIDTH, HEIGHT)  # Normalizing the canvas
        ctx.rectangle(0, 0, 1, 1)
        ctx.set_source_rgb(0, 0, 0)
        ctx.fill()
        ctx.translate(0.5, 0.5)  # Changing the current transformation matrix

        draw_radial_waveform(l, ctx, radius=(L_RADIUS))
        draw_radial_waveform(right[fftIndex], ctx, radius=(R_RADIUS), radial_offset=math.pi/4)
        ctx.arc(0, 0, C_RADIUS, 0, 2 * math.pi)
        ctx.set_source_rgb(1, 1, 1)
        ctx.stroke()

        this_title = '{}/{}.png'.format(SCRATCH_DIR, fftIndex)
        surface.write_to_png(this_title)  # Output to PNG
        paths.append(this_title)

        bar.update(fftIndex)

    if VERBOSE:
        bar.finish()
        print('DRAWING FRAMES FINISHED')

    return paths

def draw_radial_waveform(fft, context, radius=0.3, variance=0.2, radial_offset=0):
    for bucketIndex, bucketVal in enumerate(fft):
        thisAngle = radial_offset + bucketIndex * 2 * math.pi / fft.size
        thisRadius = radius + variance * abs(bucketVal)
        x = thisRadius * math.cos(thisAngle)
        y = thisRadius * math.sin(thisAngle)

        context.move_to(x, y) if bucketIndex == 0 else context.line_to(x, y)

    context.close_path()

    context.set_source_rgb(1, 1, 1)  # Solid color
    context.set_line_width(0.01)
    context.stroke()

def parse_audio(audio_path, big_bumps = False):
    if VERBOSE:
        print("PARSING AUDIO")

    y, sr = librosa.load(audio_path, mono=False)

    window_frame_duration = round(sr / FRAMERATE)     # @30fps

    # TODO:  Conditionally handle mono.
    l_stft = librosa.stft(y[0], n_fft=window_frame_duration, hop_length=window_frame_duration)
    r_stft = librosa.stft(y[1], n_fft=window_frame_duration, hop_length=window_frame_duration)

    #  NORMALIZE THE VALUES
    l_stft /= abs(np.max(np.abs(l_stft), axis=(0 if big_bumps else None)))
    r_stft /= abs(np.max(np.abs(r_stft), axis=(0 if big_bumps else None)))

    if VERBOSE:
        print("AUDIO FINISHED")

    return (np.transpose(l_stft), np.transpose(r_stft))

def write_video(video_source, audio_source, fps=FRAMERATE, size=None, is_color=True, format="X264", destination='out.mp4'):
    if VERBOSE:
        print("WRITING VIDEO")

    if os.path.exists(destination):
        os.remove(destination)

    os.system('ffmpeg -framerate {} -i {} -i {} -c:v libx264 -c:a copy -tune animation -pix_fmt yuv420p {}'.format(fps, video_source, audio_source, destination))

    if VERBOSE:
        print("VIDEO FINISHED")


def reset_dir(dir):
    if os.path.exists(dir):
        shutil.rmtree(dir)

    os.mkdir(dir)

def SOTA_audio_to_video(audio_url, outfile='video.mp4'):
    if VERBOSE:
        print("DOWNLOADING AUDIO: " + audio_url)

    reset_dir(DOWNLOAD_DIR)

    audio_path = DOWNLOAD_DIR + '/' + audio_url.split('/')[-1]
    urllib.request.urlretrieve(audio_url, audio_path)

    if not os.path.exists(audio_path):
        raise FileNotFoundError

    if VERBOSE:
        print("AUDIO DOWNLOADED")

    l, r = parse_audio(audio_path)
    draw_stereo_output(l, r)
    write_video(audio_path, destination=(OUTPUT_DIR + "/" + outfile))

def SOTA_audio_to_video2():
    AUDIO_LEFT_CHANNEL, AUDIO_RIGHT_CHANNEL = parse_audio('./billiejean.mp3')
    y, sr = librosa.load('./billiejean.mp3')
    duration = librosa.get_duration(y, sr=sr)

    animation = editor.VideoClip(make_frame=make_frame, duration=duration)


# SOTA_audio_to_video('http://rss.art19.com/episodes/b6afde43-618d-409f-af43-4dc42699af08.mp3')
SOTA_audio_to_video2()

# parser = argparse.ArgumentParser(description='Convert an audio file into an animated video for the SOTA (State of the Art) Podcast.')
# parser.add_argument(['-u', '--url'], action='store', )