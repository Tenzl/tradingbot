FROM python:3.11-slim

WORKDIR /app

# Copy requirements
COPY trading_bot_realtime.py .
COPY xgb_realtime_bot_model.json .

# Install dependencies
RUN pip install --no-cache-dir \
    pandas numpy xgboost requests \
    && apt-get update \
    && apt-get install -y wget build-essential \
    && wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz \
    && tar -xzf ta-lib-0.4.0-src.tar.gz \
    && cd ta-lib/ \
    && ./configure --prefix=/usr \
    && make \
    && make install \
    && cd .. \
    && rm -rf ta-lib ta-lib-0.4.0-src.tar.gz \
    && pip install TA-Lib \
    && apt-get clean

# Run bot
CMD ["python", "-u", "trading_bot_realtime.py"]
