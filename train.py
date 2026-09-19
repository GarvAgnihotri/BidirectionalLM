import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import os
import json
import math
import glob
import time
from model import BidirectionalLM
from dataset import build_dataloaders


training_config = {
    "sequence_length": 512,
    "mask_probability": 0.15,
    "batch_size": 32,
    "number_of_epochs": 20,
    "learning_rate": 1e-4,
    "warmup_steps": 10000,
    "weight_decay": 0.01,
    "gradient_clip": 1.0,
    "save_every_n_steps": 500,
    "log_every_n_steps": 100,
    "checkpoint_folder": "checkpoints",
    "max_training_hours": 4,
}

model_config = {
    "embedding_size": 768,
    "number_of_layers": 12,
    "number_of_heads": 12,
    "feedforward_size": 3072,
    "dropout": 0.1,
}


def warmup_then_cosine(current_step, warmup_steps, total_steps):
    if current_step < warmup_steps:
        return current_step / warmup_steps
    progress = (current_step - warmup_steps) / (total_steps - warmup_steps)
    return 0.5 * (1 + math.cos(math.pi * progress))


def find_latest_checkpoint(checkpoint_folder):
    all_checkpoints = glob.glob(os.path.join(checkpoint_folder, "checkpoint_step_*.pt"))
    if not all_checkpoints:
        return None
    latest = max(all_checkpoints, key=lambda path: int(path.split("_step_")[1].replace(".pt", "")))
    return latest


def save_checkpoint(model, optimizer, scheduler, epoch, global_step, training_history, folder):
    os.makedirs(folder, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "global_step": global_step,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "training_history": training_history,
        "training_config": training_config,
        "model_config": model_config,
    }

    checkpoint_path = os.path.join(folder, f"checkpoint_step_{global_step}.pt")
    torch.save(checkpoint, checkpoint_path)

    latest_path = os.path.join(folder, "latest.pt")
    torch.save(checkpoint, latest_path)

    print(f"Checkpoint saved → step {global_step} ({checkpoint_path})")


def load_checkpoint(checkpoint_path, model, optimizer, scheduler):
    print(f"Resuming from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    scheduler.load_state_dict(checkpoint["scheduler_state"])

    resumed_epoch = checkpoint["epoch"]
    resumed_step = checkpoint["global_step"]
    resumed_history = checkpoint.get("training_history", [])

    print(f"Resumed from epoch {resumed_epoch + 1}, step {resumed_step}")
    return resumed_epoch, resumed_step, resumed_history


def evaluate(model, validation_loader, vocabulary_size, device, max_batches=100):
    model.eval()
    loss_function = nn.CrossEntropyLoss(ignore_index=-100)
    total_loss = 0
    total_batches = 0

    with torch.no_grad():
        for batch_index, batch in enumerate(validation_loader):
            if batch_index >= max_batches:
                break
            masked_tokens = batch["masked_tokens"].to(device)
            target_tokens = batch["target_tokens"].to(device)
            token_logits = model(masked_tokens)
            loss = loss_function(token_logits.view(-1, vocabulary_size), target_tokens.view(-1))
            total_loss += loss.item()
            total_batches += 1

    model.train()
    return total_loss / max(total_batches, 1)


def print_session_summary(session_start_step, global_step, session_start_time):
    steps_this_session = global_step - session_start_step
    minutes_elapsed = (time.time() - session_start_time) / 60
    print("\n" + "="*50)
    print("SESSION SUMMARY")
    print(f"  Steps this session : {steps_this_session}")
    print(f"  Total steps done   : {global_step}")
    print(f"  Time elapsed       : {minutes_elapsed:.1f} minutes")
    print(f"  To resume later    : python train.py")
    print("="*50 + "\n")


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    max_seconds = training_config["max_training_hours"] * 3600
    session_start_time = time.time()

    print("Loading WikiText-103...")
    train_loader, validation_loader, vocabulary_size, mask_token_id, tokenizer = build_dataloaders(
        sequence_length=training_config["sequence_length"],
        mask_probability=training_config["mask_probability"],
        batch_size=training_config["batch_size"],
    )

    total_training_steps = len(train_loader) * training_config["number_of_epochs"]
    print(f"Total steps for full training: {total_training_steps}")

    model = BidirectionalLM(
        vocabulary_size=vocabulary_size,
        mask_token_id=mask_token_id,
        **model_config
    ).to(device)

    print(f"Model parameters: {model.count_parameters() / 1e6:.1f}M")

    optimizer = AdamW(
        model.parameters(),
        lr=training_config["learning_rate"],
        weight_decay=training_config["weight_decay"],
        betas=(0.9, 0.999)
    )

    scheduler = CosineAnnealingLR(optimizer, T_max=total_training_steps)
    loss_function = nn.CrossEntropyLoss(ignore_index=-100)

    start_epoch = 0
    global_step = 0
    training_history = []

    latest_checkpoint = find_latest_checkpoint(training_config["checkpoint_folder"])
    if latest_checkpoint:
        start_epoch, global_step, training_history = load_checkpoint(
            latest_checkpoint, model, optimizer, scheduler
        )
    else:
        print("No checkpoint found — starting fresh training")

    session_start_step = global_step
    time_limit_hit = False

    for epoch in range(start_epoch, training_config["number_of_epochs"]):
        model.train()
        epoch_loss = 0
        batches_processed = 0

        for batch in train_loader:
            elapsed_seconds = time.time() - session_start_time
            if elapsed_seconds >= max_seconds:
                print(f"\nTime limit of {training_config['max_training_hours']} hours reached!")
                save_checkpoint(model, optimizer, scheduler, epoch, global_step, training_history,
                               training_config["checkpoint_folder"])
                print_session_summary(session_start_step, global_step, session_start_time)
                time_limit_hit = True
                break

            masked_tokens = batch["masked_tokens"].to(device)
            target_tokens = batch["target_tokens"].to(device)

            current_lr = training_config["learning_rate"] * warmup_then_cosine(
                global_step, training_config["warmup_steps"], total_training_steps
            )
            for param_group in optimizer.param_groups:
                param_group["lr"] = current_lr

            token_logits = model(masked_tokens)
            loss = loss_function(token_logits.view(-1, vocabulary_size), target_tokens.view(-1))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), training_config["gradient_clip"])
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            batches_processed += 1
            global_step += 1

            if global_step % training_config["log_every_n_steps"] == 0:
                average_loss = epoch_loss / batches_processed
                elapsed_minutes = (time.time() - session_start_time) / 60
                remaining_minutes = (max_seconds - (time.time() - session_start_time)) / 60
                print(f"Epoch {epoch+1} | Step {global_step} | Loss {average_loss:.4f} | LR {current_lr:.2e} | Elapsed {elapsed_minutes:.0f}min | Remaining {remaining_minutes:.0f}min")
                training_history.append({"step": global_step, "loss": average_loss, "lr": current_lr})

            if global_step % training_config["save_every_n_steps"] == 0:
                save_checkpoint(model, optimizer, scheduler, epoch, global_step, training_history,
                               training_config["checkpoint_folder"])

        if time_limit_hit:
            break

        validation_loss = evaluate(model, validation_loader, vocabulary_size, device)
        print(f"Epoch {epoch+1} complete | Validation Loss: {validation_loss:.4f}")

        save_checkpoint(model, optimizer, scheduler, epoch + 1, global_step, training_history,
                       training_config["checkpoint_folder"])

    if not time_limit_hit:
        print("\nFull training complete!")
        print_session_summary(session_start_step, global_step, session_start_time)

    with open("training_history.json", "w") as history_file:
        json.dump(training_history, history_file, indent=2)


if __name__ == "__main__":
    train()
