import torch
import torch.nn.functional as F
from model import BidirectionalLM


def load_model(checkpoint_path, vocabulary_size, mask_token_id, device, model_config):
    model = BidirectionalLM(
        vocabulary_size=vocabulary_size,
        mask_token_id=mask_token_id,
        **model_config
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    return model


def generate_bidirectional(
    model,
    tokenizer,
    prompt_text,
    total_length,
    confidence_threshold_start,
    confidence_threshold_end,
    number_of_steps,
    device
):
    prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=False)
    number_of_masks = total_length - len(prompt_tokens)

    token_ids = prompt_tokens + [tokenizer.mask_token_id] * number_of_masks
    token_ids = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0).to(device)

    mask_positions = set(range(len(prompt_tokens), total_length))

    generation_steps = []

    for step_number in range(number_of_steps):
        if not mask_positions:
            break

        with torch.no_grad():
            token_logits = model(token_ids)

        token_probabilities = F.softmax(token_logits, dim=-1)
        confidence_scores, predicted_tokens = token_probabilities.max(dim=-1)

        step_progress = step_number / number_of_steps
        current_threshold = confidence_threshold_start + (confidence_threshold_end - confidence_threshold_start) * step_progress

        left_cursor = min(mask_positions)
        right_cursor = max(mask_positions)

        left_confidence = confidence_scores[0, left_cursor].item()
        right_confidence = confidence_scores[0, right_cursor].item()

        tokens_revealed_this_step = []

        for position in list(mask_positions):
            position_confidence = confidence_scores[0, position].item()
            is_left_side = position <= (left_cursor + right_cursor) // 2
            is_right_side = not is_left_side

            if position_confidence >= current_threshold:
                token_ids[0, position] = predicted_tokens[0, position]
                mask_positions.discard(position)
                tokens_revealed_this_step.append({
                    "position": position,
                    "token": tokenizer.decode([predicted_tokens[0, position].item()]),
                    "confidence": round(position_confidence, 3),
                    "side": "left" if is_left_side else "right"
                })

        if not tokens_revealed_this_step and mask_positions:
            most_confident_position = max(mask_positions, key=lambda pos: confidence_scores[0, pos].item())
            token_ids[0, most_confident_position] = predicted_tokens[0, most_confident_position]
            most_confident_score = confidence_scores[0, most_confident_position].item()
            tokens_revealed_this_step.append({
                "position": most_confident_position,
                "token": tokenizer.decode([predicted_tokens[0, most_confident_position].item()]),
                "confidence": round(most_confident_score, 3),
                "side": "forced"
            })
            mask_positions.discard(most_confident_position)

        current_sequence = tokenizer.decode(token_ids[0].tolist(), skip_special_tokens=False)
        generation_steps.append({
            "step": step_number + 1,
            "revealed_tokens": tokens_revealed_this_step,
            "current_sequence": current_sequence,
            "masks_remaining": len(mask_positions)
        })

        print(f"\nStep {step_number + 1} (threshold={current_threshold:.2f}):")
        for revealed in tokens_revealed_this_step:
            side_label = f"[{revealed['side'].upper()}]"
            print(f"  {side_label} position {revealed['position']}: '{revealed['token']}' (confidence={revealed['confidence']})")
        print(f"  Sentence so far: {tokenizer.decode(token_ids[0].tolist(), skip_special_tokens=True)}")

    final_text = tokenizer.decode(token_ids[0].tolist(), skip_special_tokens=True)
    return final_text, generation_steps


def demonstrate_generation(model, tokenizer, device):
    print("\n" + "="*60)
    print("BIDIRECTIONAL CONFIDENCE-GUIDED GENERATION DEMO")
    print("="*60)

    prompt = "My name is"

    final_text, steps = generate_bidirectional(
        model=model,
        tokenizer=tokenizer,
        prompt_text=prompt,
        total_length=20,
        confidence_threshold_start=0.95,
        confidence_threshold_end=0.70,
        number_of_steps=15,
        device=device
    )

    print(f"\nFinal generated text:\n{final_text}")
    print(f"\nTotal generation steps: {len(steps)}")

    return final_text


if __name__ == "__main__":
    from transformers import AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    model_config = {
        "embedding_size": 768,
        "number_of_layers": 12,
        "number_of_heads": 12,
        "feedforward_size": 3072,
        "dropout": 0.1,
    }

    model = load_model(
        checkpoint_path="checkpoints/checkpoint_final.pt",
        vocabulary_size=tokenizer.vocab_size,
        mask_token_id=tokenizer.mask_token_id,
        device=device,
        model_config=model_config
    )

    demonstrate_generation(model, tokenizer, device)
