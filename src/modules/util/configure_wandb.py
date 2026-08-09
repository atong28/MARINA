import os
import json
import wandb

from ..log import get_logger, setup_file_logging, is_main_process

from ..core.const import WANDB_API_KEY_FILE
from ..marina.args import MARINAArgs
from ..spectre.args import SPECTREArgs

# Get logger after imports to avoid circular dependency
logger = get_logger(__file__)


def configure_wandb(args: MARINAArgs | SPECTREArgs, results_path: str, today: str,
                    log_dir: str | None = None):
    """_summary_

    Args:
        args (MARINAArgs | SPECTREArgs): _description_
        results_path (str): _description_
        today (str): _description_

    Raises:
        RuntimeError: _description_

    Returns:
        _type_: _description_
    """
    experiment_id = f"{args.experiment_name}_{today}"

    # Where logs.txt and params.json go. On Nautilus results_path is an emptyDir, so a
    # preempted run loses its log exactly when you need it to find out what happened;
    # log_dir lets the caller point them at persistent storage instead.
    log_dir = log_dir or results_path

    if is_main_process() and args.train:
        os.makedirs(results_path, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        setup_file_logging(logger, os.path.join(log_dir, "logs.txt"))
        logger.info("[Main] Parsed args:\n%s", args)
        if log_dir != results_path:
            logger.info("[Main] Logging to persistent path: %s", log_dir)

        with open(os.path.join(log_dir, "params.json"), "w") as fp:
            json.dump(vars(args), fp, indent=2)

        # SDSC compute nodes have no outbound network, so the chain runs WANDB_MODE=offline
        # and the login node syncs the run directory afterwards. login() needs the network,
        # and offline runs never authenticate, so the key file is not required there.
        if os.environ.get('WANDB_MODE') == 'offline':
            logger.info('[Main] WANDB_MODE=offline; skipping login, sync from a login node.')
        else:
            # WANDB_API_KEY_FILE is None on env-var setups, which os.path.exists rejects
            # with a TypeError rather than the intended error message.
            if not WANDB_API_KEY_FILE or not os.path.exists(WANDB_API_KEY_FILE):
                raise RuntimeError(
                    f"WANDB API key file not found at {WANDB_API_KEY_FILE}")

            with open(WANDB_API_KEY_FILE) as kf:
                key = json.load(kf)["key"]

            wandb.login(key=key)

        # resume="allow" only resumes when an explicit id is given; without one wandb
        # mints a fresh id every time, so each chunk of a chained run would land in its
        # own W&B run. experiment_id is stable across chunks exactly when the launcher
        # pins SMART_RUN_ID, which is the same condition that makes the results
        # directory stable -- so tying the run id to it keeps the two in step.
        init_kwargs = {}
        if getattr(args, "resume", False):
            init_kwargs["id"] = experiment_id

        wandb_run = wandb.init(
            project=args.project_name,
            name=experiment_id,
            config=vars(args),
            resume="allow",
            **init_kwargs,
        )
    else:
        # ensure path exists before creating a logger
        if is_main_process():
            os.makedirs(log_dir, exist_ok=True)
            setup_file_logging(logger, os.path.join(log_dir, "logs.txt"))

        wandb_run = None

    return wandb_run
