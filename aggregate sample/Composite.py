from flask import Flask
from flask import redirect, session, url_for, render_template 
import asyncio
import aiohttp
import time
import requests


app = Flask(__name__)


def scomposition(site):
    with requests.get(site) as resp:
        print('Read {} from {}'.format(resp.text, site))
    return


@app.route("/sync_composition")
def sync_composition():
    sites = [
        "http://127.0.0.1:8001/patient",
        "http://127.0.0.1:8002/aichat",
        "http://127.0.0.1:8003/apointment"
    ]
    for site in sites:
        scomposition(site)
    return 'Hello World!'


async def acomposition(site):
    async with aiohttp.ClientSession() as session:
        async with session.get(site) as resp:
            print('Read {} from {}'.format(resp.content_length, site))


@app.route("/async_composition")
async def async_composition():
    sites = [
        "http://127.0.0.1:8001/patient",
        "http://127.0.0.1:8002/aichat",
        "http://127.0.0.1:8003/apointment"
    ]
    # tasks = asyncio.create_task([acomposition(site) for site in sites])
    tasks = [acomposition(site) for site in sites]
    await asyncio.gather(*tasks)
    return 'Hello World!'


if __name__ == "__main__":
    app.run(port="8000")
