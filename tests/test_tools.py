import os
import time

from src.tools import (
    get_stock_price,
    download_youtube_audio_as_mp3,
    convert_from_mp3_to_wav,
    get_wav_transcript,
    get_most_recent_reddit_submissions,
    get_current_temperature,
)


def test_get_stock_price():
    max_elapsed_time = 3

    start_time = time.time()
    quote = get_stock_price(ticker="APPL")
    end_time = time.time()
    float(quote)
    assert end_time - start_time < max_elapsed_time

    start_time = time.time()
    quote = get_stock_price(ticker="Apple")
    end_time = time.time()
    assert (
        quote == "Apple is not a valid ticker"
        and end_time - start_time < max_elapsed_time
    )


def test_download_youtube_audio_as_mp3():
    max_elapsed_time = 5

    start_time = time.time()
    filename_mp3 = download_youtube_audio_as_mp3(
        url="https://www.youtube.com/watch?v=n9R1ts_xfgc", output_dir="trash"
    )
    end_time = time.time()

    os.remove(filename_mp3)
    os.rmdir("/".join(filename_mp3.split("/")[:-1]))

    assert filename_mp3.endswith("/NVIDIA GTC 2024 Keynote Teaser.mp3")
    assert end_time - start_time < max_elapsed_time


def test_convert_from_mp3_to_wav():
    max_elapsed_time = 3

    filename_mp3 = download_youtube_audio_as_mp3(
        url="https://www.youtube.com/watch?v=n9R1ts_xfgc", output_dir="trash"
    )

    start_time = time.time()
    filename_wav = convert_from_mp3_to_wav(filename_mp3)
    end_time = time.time()

    os.remove(filename_mp3)
    os.remove(filename_wav)
    os.rmdir("/".join(filename_mp3.split("/")[:-1]))

    assert filename_wav.endswith("/NVIDIA GTC 2024 Keynote Teaser - 300s.wav")
    assert end_time - start_time < max_elapsed_time


def test_get_wav_transcript():
    max_elapsed_time = 3

    filename_mp3 = download_youtube_audio_as_mp3(
        url="https://www.youtube.com/watch?v=n9R1ts_xfgc", output_dir="trash"
    )
    filename_wav = convert_from_mp3_to_wav(filename_mp3)

    start_time = time.time()
    transcript = get_wav_transcript(filename_wav)
    end_time = time.time()

    os.remove(filename_mp3)
    os.remove(filename_wav)
    os.rmdir("/".join(filename_mp3.split("/")[:-1]))

    assert transcript == (
        " Ladies and gentlemen, please welcome Jensen Huang. "
        "The purpose of GTC is to inspire the world on the art of the possible of accelerated computing. "
        "Have a great GTC. [Music]"
    )
    assert end_time - start_time < max_elapsed_time


def test_get_most_recent_reddit_submissions():
    import os

    client_id = os.environ.get("REDDIT_CLIENT_ID", None)
    client_secret = os.environ.get("REDDIT_CLIENT_SECRET", None)
    user_agent = os.environ.get("REDDIT_USER_AGENT", None)

    text = get_most_recent_reddit_submissions(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent,
        subreddit_name="LocalLLaMA",
    )
    assert text.count("New post from ") == 10

    text = get_most_recent_reddit_submissions(
        client_id="dummy",
        client_secret="dummy",
        user_agent="dummy",
        subreddit_name="LocalLLaMA",
    )
    assert text == "Failed authentification"


def test_get_current_temperature():
    import os

    openweathermap_api_key = os.environ.get("OPENWEATHERMAP_API_KEY", None)

    t_celsius = get_current_temperature(openweathermap_api_key)
    assert -20 < float(t_celsius) < 40
