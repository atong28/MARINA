# MARINA Repository

## Website Setup

Ensure `docker` and `docker-compose` are installed, and copy `.env.example` to `.env`, and configure the variables as you see fit. You must absolutely set `MODEL_DATA_DIR`, and by default the website will run on cpu inference. Note that some legacy machines may be buggy with numpy, so if there is a repeated import error then set `LEGACY_NUMPY=true`.

Download the model (run it in a environment with `gdown` installed, it is installed if you download `pixi` and install the environment below)
```bash
bash download_model.sh
```

Start the docker containers:
```bash
bash start.sh
```

## Code Installation

Install pixi according to the following instructions:
```
https://pixi.sh/dev/installation/
```
If you are not running on linux-64, you can try adding your distro into `pixi.toml` and install anyways, but no guarantees for support. Running the following command should automatically boot you into the shell with the loaded environment:
```
pixi shell
```
To just install the environment, use
```
pixi i
```



