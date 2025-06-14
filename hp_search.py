"""
hp_search.py

Perform hyperparameter tuning for the VAE training pipeline using Optuna.

Workflow:
  1. Parse hyperparameter-search arguments (number of trials, core train_vae flags).
  2. Optionally stage Zarr data locally once for all trials.
  3. Define an Optuna `objective()` that:
     - Samples key hyperparameters (lr, weight‐decay, latent_dim, slow_weight, etc.).
     - Assembles CLI args for a short training run via `train_and_evaluate`.
     - Returns the validation loss.
  4. Create or resume an Optuna study (SQLite-backed) and optimize for N trials.
  5. Persist the best hyperparameters to `best_hparams.json` and the study DB.
  6. Optionally send an email notification with the results.
"""

import json
import argparse
import optuna
import logging
from pymaap import init_general_logger
init_general_logger()
import tempfile
from pathlib import Path
import smtplib
from email.mime.text import MIMEText

from train_vae import (
	parse_args as train_parse_args, 
	train_and_evaluate, 
	stage_zarr_locally
)

def parse_args():
    """
    Parse command-line arguments for the hyperparameter search itself.

    Required:
      --trials            number of Optuna trials to run
      -b/--bucket         S3 bucket name
      -f/--folder         S3 prefix for Zarr stores
      -tr/--train-geohashes
      -va/--val-geohashes
      -te/--test-geohashes (optional)
      -o/--out            output directory for study DB and logs
      -A/--aws-creds      JSON file with AWS credentials

    Optional:
      --batch-size, --patch-size, --num-workers,
      --stage-zarr / --no-stage-zarr,
      --trial-epochs, --trial-patience

    Returns:
      Namespace of parsed hyperparameter-search args.
    """
    p = argparse.ArgumentParser(description="Hyperparameter search for VAE")
    p.add_argument("--trials", type=int, default=20,
                   help="How many Optuna trials to run")
    # collect *just* the core training flags, the rest are handed off directly
    p.add_argument("-b","--bucket",       required=True)
    p.add_argument("-f","--folder",       required=True)
    p.add_argument("-tr","--train-geohashes", required=True)
    p.add_argument("-va","--val-geohashes",   required=True)
    p.add_argument("-te","--test-geohashes",  default="")
    p.add_argument("-o","--out",          required=True)
    p.add_argument("-A","--aws-creds",    required=True)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--patch-size", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--stage-zarr", action="store_true",
                   help="Global: download Zarrs locally")
    p.add_argument("--no-stage-zarr", action="store_true", dest="no_stage_zarr",
                   help="Global: do NOT download Zarrs locally")
    p.add_argument("--trial-epochs", type=int, default=5,
                   help="Number of epochs per trial (default=5)")
    p.add_argument("--trial-patience", type=int, default=3,
                   help="Early stopping patience per trial (default=3)")
    hp_args_, _ = p.parse_known_args()
    return hp_args_

def send_email_notification(creds_path: Path, subject: str, body: str):
    """
    Reads SMTP credentials from creds_path (JSON), and sends an email.

    creds_path JSON structure (using gmail as the example):
      {
        "smtp_host": "smtp.gmail.com",
        "smtp_port": 587,
        "smtp_user": "youremail@gmail.com",
        "smtp_pass": "your-app-specific-password",
        "notify_to": "youremail@gmail.com"
      }

    Constructs a MIMEText message with 'subject' and 'body', logs any failures.
    """
    try:
        with open(creds_path, "r") as f:
            email_cfg = json.load(f)
    except Exception as e:
        logging.warning(f"Could not read SMTP credentials from {creds_path}: {e}")
        return

    smtp_host = email_cfg.get("smtp_host")
    smtp_port = int(email_cfg.get("smtp_port", 587))
    smtp_user = email_cfg.get("smtp_user")
    smtp_pass = email_cfg.get("smtp_pass")
    notify_to = email_cfg.get("notify_to", smtp_user)

    if not (smtp_host and smtp_port and smtp_user and smtp_pass):
        logging.warning("Missing one of smtp_host/smtp_port/smtp_user/smtp_pass in credentials. Skipping email.")
        return

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"]    = smtp_user
    msg["To"]      = notify_to

    try:
        server = smtplib.SMTP(smtp_host, smtp_port)
        server.ehlo()
        server.starttls()
        server.login(smtp_user, smtp_pass)
        server.sendmail(smtp_user, [notify_to], msg.as_string())
        server.quit()
        logging.info(f"Notification email sent to {notify_to}")
    except Exception as e:
        logging.warning(f"Failed to send email notification: {e}")

def main():
    """
    Entry point for hyperparameter search.

    1. Parse HP-search args via `parse_args()`.
    2. Load AWS credentials.
    3. Build the “base” CLI args for train_vae.
    4. If requested, stage all Zarrs locally under a single root.
    5. Create an Optuna study (SQLite at `out/hp_search.db`) or resume.
    6. Optimize the `objective()` for `trials` iterations.
    7. On completion (or interruption), log & save the best trial’s params.
    8. If email creds are found at `secrets/email_creds.json`, send a summary notification.
    """
    hp_args = parse_args()
    logging.info(f"Starting hyperparameter search with these args: {hp_args}")

    # 1) load creds once
    aws_creds = json.load(open(hp_args.aws_creds))

    # 2) build base args for train_vae
    base_train_args = [
        "-b", hp_args.bucket,
        "-f", hp_args.folder,
        "-tr", hp_args.train_geohashes,
        "-va", hp_args.val_geohashes,
        "-o", hp_args.out,
        "-A", hp_args.aws_creds,
    ]
    if hp_args.test_geohashes:
        base_train_args += ["-te", hp_args.test_geohashes]

    # 3) Make a single staging root if requested.  All geohashes go under `stage_root/`.
    if hp_args.stage_zarr and not hp_args.no_stage_zarr:
        stage_root = tempfile.mkdtemp(prefix="zarr_hp_")
        all_ghs = (hp_args.train_geohashes.split(",") +
                   hp_args.val_geohashes.split(",") +
                   (hp_args.test_geohashes.split(",") if hp_args.test_geohashes else []))
        for gh in all_ghs:
            s3uri = f"s3://{hp_args.bucket}/{hp_args.folder}/{gh}/{gh}_zarr"
            # PASS parent_dir=stage_root → THIS ensures all subfolders live under stage_root
            local_path = stage_zarr_locally(s3uri, aws_creds, parent_dir=stage_root)
            logging.info(f"Staged {gh} at {local_path}")
        # Now point every trial at that one folder:
        base_train_args += ["--stage-zarr-root", stage_root]
        # otherwise, if not staging, we'll stream from S3 for each trial (no change to base_train_args).

    trial_epochs = hp_args.trial_epochs
    trial_patience = hp_args.trial_patience

    def objective(trial):
        """
        Optuna objective function for a single trial.

        1. Samples hyperparameters (lr, weight_decay, latent_dim, slow_weight,
        scheduler_gamma, hard_power, max_step, num_time_pairs, time_gap_exp).
        2. Builds a modified CLI “cmd” list by injecting these into the base train arguments.
        3. Parses into a Namespace via `train_parse_args(cmd)`.
        4. Calls `train_and_evaluate(args)` for a short run (trial_epochs, trial_patience).
        5. Returns the resulting validation loss to be minimized by Optuna.
        """
        # 1) sample
        lr           = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-7,1e-3, log=True)
        latent_dim   = trial.suggest_int("latent_dim",16,256,log=True)
        slow_weight  = trial.suggest_float("slow_weight",0.0,1.0)
        scheduler_gamma = trial.suggest_float("scheduler_gamma",0.1,0.9)
        hard_power   = trial.suggest_float("hard_power", 1.0, 5.0) # 1.0 = uniform, 5.0 = very aggressive
        max_step     = trial.suggest_int("max_step", 1, 180) # search integer between 1 and X
        num_time_pairs  = trial.suggest_float("num_time_pairs", 0.1, 1.0)
        time_gap_exp = trial.suggest_float("time_gap_exp", 0.5, 2.0)

        # 2) build base + these hparams
        cmd = base_train_args.copy()
        cmd += [
            "--lr", str(lr),
            "--weight-decay", str(weight_decay),
            "--latent-dim", str(latent_dim),
            "--slow-weight", str(slow_weight),
            "--scheduler-gamma", str(scheduler_gamma),
            "--hard-power", str(hard_power),
            "--epochs", str(trial_epochs),
            "--early-stopping", str(trial_patience),
            "--max-step", str(max_step),
            "--num-time-pairs", str(num_time_pairs),
            "--time-gap-exp", str(time_gap_exp)
        ]

        # optional overrides from hp_args
        if hp_args.batch_size is not None:
            cmd += ["--batch-size", str(hp_args.batch_size)]
        if hp_args.patch_size is not None:
            cmd += ["--patch-size", str(hp_args.patch_size)]
        if hp_args.num_workers is not None:
            cmd += ["--num-workers", str(hp_args.num_workers)]
        # Only add “--stage-zarr” if we did NOT already give “--stage-zarr-root”
        if hp_args.stage_zarr and "--stage-zarr-root" not in base_train_args:
           cmd += ["--stage-zarr"]
        if hp_args.no_stage_zarr:
            cmd += ["--no-stage-zarr"]

        # run
        args = train_parse_args(cmd)
        val_loss, _ = train_and_evaluate(args)
        return val_loss

    # calculate global min/max vals for dataset normalization
    all_ghs = (
        hp_args.train_geohashes.split(",")
        + hp_args.val_geohashes.split(",")
        + (hp_args.test_geohashes.split(",") if hp_args.test_geohashes else [])
    )

    # persist study to disk (SQLite) so that we can resume and inspect later
    db_path = Path(hp_args.out) / "hp_search.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    # launch the study
    study = optuna.create_study(
        storage=f"sqlite:///{db_path}",
        study_name="vae_hp_search",
        load_if_exists=True,
        direction="minimize",
        sampler=optuna.samplers.TPESampler())

    # If you redirect Optuna’s logs to this folder, you’ll also get an “optuna.log” there.
    optuna.logging.get_logger("optuna").addHandler(
        logging.FileHandler(Path(hp_args.out)/"optuna.log")
    )

    try:
        study.optimize(objective, n_trials=hp_args.trials)
    except KeyboardInterrupt:
        # Allow CTRL+C to stop and still save partial progress
        logging.warning("Hyperparameter search interrupted by user. Saving progress so far...")

    # print and save best parameters
    try:
        best = study.best_trial # may raise ValueError if no trial was stored
        logging.info("Best trial:")
        logging.info(f"  Val loss: {best.value:.4f}")
        for name, val in study.best_trial.params.items():
            logging.info(f"  {name}: {val}")
        with open(Path(hp_args.out) / "best_hparams.json", "w") as f:
            json.dump(study.best_trial.params, f, indent=2)
    except ValueError:
        logging.warning("No completed trial found; best_trial does not exist.")

    # send email notification if credentials exist
    # credentials are expected in secrets/email_creds.json
    email_creds_path = Path("secrets/email_creds.json")
    log_location = Path(hp_args.out) / 'optuna.log'
    if email_creds_path.exists():
        subj = f"Optuna HP Search Complete: best val_loss={study.best_trial.value:.4f}"
        body = (
            f"Optuna hyperparameter search has finished.\n\n"
            f"Hyperparameter search args: {hp_args}"
            f"Best trial value: {study.best_trial.value:.4f}\n"
            f"Best params:\n{json.dumps(study.best_trial.params, indent=2)}\n\n"
            f"Full study database: {db_path}\n"
            f"Optuna log: {log_location}\n"
        )
        send_email_notification(email_creds_path, subj, body)
    else:
        logging.warning(f"Email credentials not found at {email_creds_path}; skipping notification.")

if __name__ == "__main__":
    main()

# ----------------------------------------------------------------------
# Examples:
#
# For a "smoke test":
#  python hp_search.py \
#    -b rgc-zarr-store \
#    -f data \
#    -tr 9vgm0,9vgm1,9vgmd \
#    -va 9vgm2 \
#    -te 9vgm6 \
#    -o results/hp_run1 \
#    -A secrets/aws_rgc-zarr-store.json \
#    --stage-zarr \
#    --batch-size 16 \
#    --patch-size 128 \
#    --trial-epochs 3 \
#    --trial-patience 2
#    --trials 3
#
# This will run 3 Optuna trials, each doing 3 epochs of training on
# geohashes [9vgm0, 9vgm1, 9vgmd], validating on 9vgm2, testing on 9vgm6, all with
# your local Zarr store staged, and caching the best hparams in best_hparams.json
# plus keeping a full record of every trial in results/hp_run1/hp_search.db.
# 
# For a full run, do something like:
# python hp_search.py \
#   -b rgc-zarr-store \
#   -f data \
#   -tr 9vgm0,9vgm1,9vgm6,9vgmd \
#   -va 9vgkg \
#   -te 9vgm2 \
#   -o results/full_hp_run \
#   -A secrets/aws_rgc-zarr-store.json \
#   --stage-zarr \
#   --batch-size 32 \
#   --patch-size 128 \
#   --trial-epochs 30 \
#   --trial-patience 5 \
#   --trials 30
#   --num-workers 0 # for running script on EC2
# 
# You can increase --trials to 50-100 or --trial-epochs to 100 if you're confident
# in the disk and GPU throughput. 
# ----------------------------------------------------------------------
