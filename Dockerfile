# Bot image. Build from the repo root: docker build -t my-bot .
# Matches run it with no network, a read-only disk (only /tmp is writable), 1 CPU, 512 MB,
# as user 65534, with the map mounted read-only at /arena. The bot uses only the standard library.
FROM python:3.13.18-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /bot
COPY mechfight.py world.py bot.py ./
# compile ahead of time so start-up is quick on the read-only disk
RUN python -m compileall -q /bot && chmod -R a+rX /bot
ENTRYPOINT ["python3", "-u", "/bot/bot.py"]
