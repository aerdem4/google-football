pip install -r requirements.txt

apt-get update -y
apt-get install -y libsdl2-gfx-dev libsdl2-ttf-dev

git clone -b v2.8 https://github.com/google-research/football.git
mkdir -p football/third_party/gfootball_engine/lib

wget https://storage.googleapis.com/gfootball/prebuilt_gameplayfootball_v2.8.so -O football/third_party/gfootball_engine/lib/prebuilt_gameplayfootball.so
cd football && GFOOTBALL_USE_PREBUILT_SO=1 pip install .

mkdir -p /kaggle_simulations/agent