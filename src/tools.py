import os
import re
import subprocess
from urllib.request import urlopen

from bs4 import BeautifulSoup
from ip2geotools.databases.noncommercial import DbIpCity
import praw
from pytube import YouTube
import requests


def get_stock_price(ticker: str) -> str:
    """
    This function queries Yahoo Finance to get the current price for a provided stock ticker.

    Args:
        ticker (str): The stock ticker e.g. AAPL for Apple.

    Returns:
        str: The current stock price.
    """
    try:
        response = requests.get(
            f"https://finance.yahoo.com/quote/{ticker}",
            headers={
                "User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/114.0"
            },
            timeout=10,
        )
    except:
        return "Unable to reach Yahoo Finance"
    if response.url != f"https://finance.yahoo.com/quote/{ticker}":
        return f"{ticker} is not a valid ticker"
    soup = BeautifulSoup(response.text, "html.parser")
    current = soup.select_one(
        f'fin-streamer[data-field="regularMarketPrice"][data-symbol="{ticker}"]'
    )
    if current is None:
        return "Invalid request"
    else:
        return current["value"]


def download_youtube_audio_as_mp3(url: str, output_dir: str = "") -> str:
    """
    This function downloads a YouTube video from a URL as mp3 and returns the path to the 
    saved mp3 file.
    For example, download_youtube_audio_as_mp3([YOUTUBE URL]) returns [PATH TO MP3 FILE].

    Args:
        url (str): The YouTube URL, e.g. https://www.youtube.com/watch?v=abcdefg.
        output_dir (str): The folder where the mp3 is to be saved. Defaults to: "".

    Returns:
        str: path to the saved mp3 file.
    """
    if url.startswith("https://www.youtube.com/watch?v=") is False:
        return "Error: url should start with https://www.youtube.com/watch?v="

    video = YouTube(url)
    audio = video.streams.filter(only_audio=True).first()
    filename = f"{video.title}.mp3"
    filename = re.sub(r"[^0-9a-zA-Z\\/@+\-:,|#().& ]+", "", filename)
    return audio.download(output_path=output_dir, filename=filename)


def convert_from_mp3_to_wav(
    filename: str, verbose: bool = False, cutoff: int = 300
) -> str:
    """
    This function converts a MP3 file to a WAV file and return the path to the saved WAV file.
    For example, convert_from_mp3_to_wav([PATH TO MP3 FILE]) returns [PATH TO WAV FILE].

    Args:
        filename (str): The MP3 filename, e.g. "/home/my_mp3.mp3"
        verbose (bool): To print verbose statements. Defaults to: false.
        cutoff (int): The audio end time. Defaults to: 300.

    Returns:
        str: path to the saved WAV file.
    """
    if filename.endswith(".mp3") is False:
        return "Error: filename should be a MP3 file."

    if os.path.isfile(filename) is False:
        return f"RuntimeError: File {filename} not found"

    if isinstance(cutoff, int) and cutoff > 0:
        output_filename = "{filename} - {cutoff}s.wav".format(
            filename=filename[:-4], cutoff=cutoff
        )
        # command = 'ffmpeg -ss 0 -t {cutoff} -y -i "{filename}" -ar 16000 -ac 1 -c:a pcm_s16le "{output_filename}"'.format(
        #     cutoff=cutoff, filename=filename, output_filename=output_filename
        # )
        command = f'ffmpeg -ss 0 -t {cutoff} -y -i "{filename}" -ar 16000 -ac 1 -c:a pcm_s16le '
        command += f'"{output_filename}"'
    else:
        output_filename = "{filename}.wav".format(filename=filename[:-4])
        # command = 'ffmpeg -y -i "{filename}" -ar 16000 -ac 1 -c:a pcm_s16le "{output_filename}"'.format(
        #     filename=filename, output_filename=output_filename
        # )
        command = f'ffmpeg -y -i "{filename}" -ar 16000 -ac 1 -c:a pcm_s16le "{output_filename}"'
    if verbose is False:
        command += " -hide_banner -loglevel error"

    subprocess.call(command, shell=True)

    return output_filename


def get_wav_transcript(filename: str, asr_model: str = "whisper") -> str:
    """
    This function returns the transcript from a WAV file.
    For example, get_wav_transcript("/home/my_wav.wav") returns "[VIDEO TRANSCRIPT]"

    Args:
        filename (str): The WAV filename, e.g. "/home/my_wav.wav".
        asr_model (str): Defaults to: "whisper".

    Returns:
        str: the transcript.
    """
    if filename.endswith(".wav") is False:
        return "Error: filename should be a WAV file."

    if os.path.isfile(filename) is False:
        # return "RuntimeError: File {filename} not found".format(filename=filename)
        return f"RuntimeError: File {filename} not found"

    if asr_model == "whisper":
        from whispercpp import Whisper

        asr_model = Whisper("base")

        result = asr_model.transcribe(filename)
        text = asr_model.extract_text(result)
        return "".join(text)
    else:
        raise ValueError


def get_most_recent_reddit_submissions(
    client_id: str, client_secret: str, user_agent: str, subreddit_name: str
) -> str:
    """
    This function returns the 10 most recent Reddit submissions in a given subreddit.

    Args:
        client_id (str): Reddit client id.
        client_secret (str): Reddit client secret.
        user_agent (str): Reddit user agent.
        subreddit_name (str): subreddit name.

    Returns:
        str: the text snippet containing the 10 most recent submissions in the given subreddit.
    """
    if subreddit_name.startswith("r/"):
        subreddit_name = subreddit_name[2:]

    reddit = praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent,
    )

    try:
        reddit.auth.scopes()
    except:
        return "Failed authentification"

    subreddit = reddit.subreddit(subreddit_name)

    combined_text = ""
    for submission in subreddit.hot(limit=10):
        combined_text += "New post from {} ({} upvotes):\nTitle: {}\n\n".format(
            submission.author.name if submission.author else "Deleted user",
            submission.ups,
            submission.title,
        )

    return combined_text[:-1]


def get_current_temperature(openweathermap_api_key: str) -> str:
    """
    This function returns the current temperature (in celsius) in the city 
    where the current IP address is located at.

    Args:
        openweathermap_api_key (str): OpenWeatherMap api key

    Returns:
        str: current temperature, in celsius.
    """
    d = str(urlopen("http://checkip.dyndns.com/").read())
    ip = re.compile(r"Address: (\d+\.\d+\.\d+\.\d+)").search(d).group(1)
    res = DbIpCity.get(ip, api_key="free")
    q = "{}, {}".format(res.city, res.country)

    complete_url = (
        "http://api.openweathermap.org/data/2.5/weather?appid="
        + openweathermap_api_key
        + "&q="
        + q
    )
    response = requests.get(complete_url)
    x = response.json()

    if x["cod"] == 200:
        t_celsius = x["main"]["temp"] - 273.15
        return str(round(t_celsius, 1))
    elif x["cod"] == 401:
        return "Invalid API key"
    else:
        return "Error with code " + x["cod"]
