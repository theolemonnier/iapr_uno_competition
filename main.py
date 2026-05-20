import argparse
import os
import math
import csv
from ultralytics import YOLO

from src.token_detection import get_player_from_centroid_position, detect_active_player


def generate_submission(model, results, output_csv):
    with open(output_csv, mode="w", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)

        # Write the header exactly as requested by Kaggle
        csv_writer.writerow(
            [
                "image_id",
                "center_card",
                "active_player",
                "player_1_cards",
                "player_2_cards",
                "player_3_cards",
                "player_4_cards",
            ]
        )

        for result in results:
            file_name = os.path.basename(result.path)
            # Remove the extension to get the exact image_id for Kaggle
            image_id = os.path.splitext(file_name)[0]

            # Extract raw box data
            raw_boxes = []
            if len(result.boxes) > 0:
                for box in result.boxes:
                    class_id = int(box.cls[0].item())
                    class_name = model.names[class_id]

                    # xywh returns
                    cx, cy = box.xywh[0][:2].tolist()
                    raw_boxes.append({"class": class_name, "x": cx, "y": cy})

            # Merge cards based on distance (450-500px) and matching class
            cards = []
            used_indices = set()

            for i in range(len(raw_boxes)):
                if i in used_indices:
                    continue

                merged = False
                for j in range(i + 1, len(raw_boxes)):
                    if j in used_indices:
                        continue

                    # Check if classes match
                    if raw_boxes[i]["class"] == raw_boxes[j]["class"]:
                        dist = math.hypot(
                            raw_boxes[i]["x"] - raw_boxes[j]["x"],
                            raw_boxes[i]["y"] - raw_boxes[j]["y"],
                        )

                        if 450 <= dist <= 500:
                            avg_x = (raw_boxes[i]["x"] + raw_boxes[j]["x"]) / 2
                            avg_y = (raw_boxes[i]["y"] + raw_boxes[j]["y"]) / 2
                            cards.append(
                                {"class": raw_boxes[i]["class"], "x": avg_x, "y": avg_y}
                            )

                            used_indices.add(i)
                            used_indices.add(j)
                            merged = True
                            break

                if not merged:
                    cards.append(
                        {
                            "class": raw_boxes[i]["class"],
                            "x": raw_boxes[i]["x"],
                            "y": raw_boxes[i]["y"],
                        }
                    )

            # Find the center card
            image_shape = result.orig_shape
            image_height, image_width = image_shape
            img_center_x, img_center_y = image_width / 2, image_height / 2

            center_card = None
            min_dist = float("inf")

            for card in cards:
                dist_to_center = math.hypot(
                    card["x"] - img_center_x, card["y"] - img_center_y
                )
                if dist_to_center < min_dist:
                    min_dist = dist_to_center
                    center_card = card

            # Extract center card class (fallback to EMPTY just in case)
            center_class = "EMPTY"
            if center_card:
                center_class = center_card["class"]
                cards.remove(center_card)

            # Assign remaining cards to players
            player_cards = {1: [], 2: [], 3: [], 4: []}

            for card in cards:
                pid = get_player_from_centroid_position(
                    (card["x"], card["y"]), image_shape
                )
                if pid in player_cards:
                    player_cards[pid].append(card["class"])

            # Format strings for the CSV
            def format_player_cards(card_list):
                return ";".join(card_list) if card_list else "EMPTY"

            p1_str = format_player_cards(player_cards[1])
            p2_str = format_player_cards(player_cards[2])
            p3_str = format_player_cards(player_cards[3])
            p4_str = format_player_cards(player_cards[4])

            # Format the Active Player correctly for Kaggle
            image_array = result.orig_img
            active_player_num = detect_active_player(image_array)
            active_player_str = f"p{active_player_num}"

            # Write the row ensuring absolutely no null/empty strings
            csv_writer.writerow(
                [
                    image_id,
                    center_class,
                    active_player_str,
                    p1_str,
                    p2_str,
                    p3_str,
                    p4_str,
                ]
            )

    print(f"Processing complete! Results saved to {output_csv}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, default="./model/best.pt")
    p.add_argument("--output_csv", type=str, default="submission.csv")
    p.add_argument(
        "--test_images_dir",
        type=str,
        default="./data/iapr-26-uno-vision-challenge/test_images",
    )
    args = p.parse_args()

    # Load trained model and predict test images
    print("Loading Best Model")
    model = YOLO(args.model_path)

    print("Predicting Test Images")
    results = model.predict(source=args.test_images_dir, save=True)

    generate_submission(model, results, args.output_csv)
