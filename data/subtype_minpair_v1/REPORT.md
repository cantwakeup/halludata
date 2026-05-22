# Subtype-Aware Symmetric Minimal-Pair Experiment Report

## 1. Repository/Data Inspection Summary
```json
{
  "coco": {
    "train_records": 82081,
    "val_records": 40137,
    "category_count": 80
  },
  "gqa": {
    "train_scene_graph": "/home/huiwei/sy/sy_data/GQA/raw/sceneGraphs/train_sceneGraphs.json",
    "val_scene_graph": "/home/huiwei/sy/sy_data/GQA/raw/sceneGraphs/val_sceneGraphs.json",
    "image_roots": [
      "/home/huiwei/sy/sy_data/GQA/raw/images/images",
      "/home/huiwei/sy/sy_data/GQA/raw/images"
    ],
    "train_records": 74289,
    "val_records": 10568,
    "train_attr_pool": 37144,
    "train_rel_pool": 37145,
    "val_attr_pool": 5284,
    "val_rel_pool": 5284
  }
}
```

## 2. Minimal-Pair Dataset Summary
# Subtype Minimal-Pair Data Report

## Counts
| split | subtype | n | yes | no | sources |
| --- | --- | --- | --- | --- | --- |
| train | cat_random | 600 | 300 | 300 | {'coco': 600} |
| train | cat_popular | 600 | 300 | 300 | {'coco': 600} |
| train | cat_hard | 600 | 300 | 300 | {'coco': 600} |
| train | attr_color | 500 | 250 | 250 | {'gqa': 500} |
| train | attr_count | 500 | 250 | 250 | {'gqa': 500} |
| train | rel_spatial | 500 | 250 | 250 | {'gqa_bbox_derived': 500} |
| train | rel_contact | 500 | 250 | 250 | {'gqa': 500} |
| val | cat_random | 200 | 100 | 100 | {'coco': 200} |
| val | cat_popular | 200 | 100 | 100 | {'coco': 200} |
| val | cat_hard | 200 | 100 | 100 | {'coco': 200} |
| val | attr_color | 200 | 100 | 100 | {'gqa': 200} |
| val | attr_count | 200 | 100 | 100 | {'gqa': 200} |
| val | rel_spatial | 200 | 100 | 100 | {'gqa_bbox_derived': 200} |
| val | rel_contact | 200 | 100 | 100 | {'gqa': 200} |

## Image Overlap
```json
{
  "cat": {
    "cat": 400,
    "attr": 0,
    "rel": 0
  },
  "attr": {
    "cat": 0,
    "attr": 691,
    "rel": 0
  },
  "rel": {
    "cat": 0,
    "attr": 0,
    "rel": 689
  }
}
```

## Label Distributions
```json
{
  "objects": {
    "shirt": 40,
    "sky": 38,
    "tree": 28,
    "wall": 26,
    "pants": 22,
    "hair": 20,
    "tail": 20,
    "grass": 18,
    "person": 18,
    "pole": 16,
    "ear": 16,
    "man": 16,
    "sign": 14,
    "trees": 14,
    "hat": 14,
    "t-shirt": 14,
    "ground": 14,
    "building": 14,
    "letters": 12,
    "car": 12
  },
  "colors": {
    "white": 174,
    "black": 106,
    "green": 104,
    "blue": 76,
    "brown": 60,
    "red": 54,
    "gray": 52,
    "orange": 28,
    "yellow": 24,
    "pink": 14,
    "purple": 8
  },
  "counts": {
    "1": 580,
    "2": 78,
    "3": 26,
    "4": 10,
    "5": 6
  },
  "relations": {
    "wearing": 470,
    "right of": 222,
    "left of": 220,
    "below": 136,
    "above": 122,
    "holding": 114,
    "sitting on": 42,
    "carrying": 20,
    "standing on": 18,
    "eating": 14,
    "lying on": 10,
    "riding": 10,
    "leaning on": 2
  },
  "negative_strategy": {
    "cooccurrence_hard_absent": 800,
    "popular_absent": 800,
    "random_absent": 800
  }
}
```

## Warnings
- None

## Examples
### cat_random
```json
[
  {
    "id": "train_coco_487114_cat_random_banana_dog_no",
    "gt_answer": "no",
    "question": "Is there a dog in the image?",
    "visual_prompt": "Question: Is there a dog in the image?\nPlease answer the question based on the image.",
    "trusted_prompt_fact": "The given image depicts the following scene: There are 7 bananas in the image. There is no dog in the image.\nPlease directly answer the following question from the image description, without guessing or reasoning.\nQuestion: Is there a dog in the image?",
    "trusted_prompt_counterfact": "The given image depicts the following scene: There are 7 bananas in the image. There is a dog in the image.\nPlease directly answer the following question from the image description, without guessing or reasoning.\nQuestion: Is there a dog in the image?"
  },
  {
    "id": "val_coco_66271_cat_random_cat_bottle_no",
    "gt_answer": "no",
    "question": "Is there a bottle in the image?",
    "visual_prompt": "Question: Is there a bottle in the image?\nPlease answer the question based on the image.",
    "trusted_prompt_fact": "The given image depicts the following scene: There is one cat in the image. There is one suitcase in the image. There is no bottle in the image.\nPlease directly answer the following question from the image description, without guessing or reasoning.\nQuestion: Is there a bottle in the image?",
    "trusted_prompt_counterfact": "The given image depicts the following scene: There is one cat in the image. There is one suitcase in the image. There is a bottle in the image.\nPlease directly answer the following question from the image description, without guessing or reasoning.\nQuestion: Is there a bottle in the image?"
  },
  {
    "id": "train_coco_363318_cat_random_sports_ball_skis_yes",
    "gt_answer": "yes",
    "question": "Is there a sports ball in the image?",
    "visual_prompt": "Question: Is there a sports ball in the image?\nPlease answer the question based on the image.",
    "trusted_prompt_fact": "The given image depicts the following scene: There are 13 chairs in the image. There are 12 persons in the image. There is one tennis racket in the image. There is a sports ball in the image.\nPlease directly answer the following question from the image description, without guessing or reasoning.\nQuestion: Is there a sports ball in the image?",
    "trusted_prompt_counterfact": "The given image depicts the following scene: There are 13 chairs in the image. There are 12 persons in the image. There is one tennis racket in the image. There is no sports ball in the image.\nPlease directly answer the following question from the image description, without guessing or reasoning.\nQuestion: Is there a sports ball in the image?"
  },
  {
    "id": "train_coco_341049_cat_random_horse_potted_plant_no",
    "gt_answer": "no",
    "question": "Is there a potted plant in the image?",
    "visual_prompt": "Question: Is there a potted plant in the image?\nPlease answer the question based on the image.",
    "trusted_prompt_fact": "The given image depicts the following scene: There is one horse in the image. There is no potted plant in the image.\nPlease directly answer the following question from the image description, without guessing or reasoning.\nQuestion: Is there a potted plant in the image?",
    "trusted_prompt_counterfact": "The given image depicts the following scene: There is one horse in the image. There is a potted plant in the image.\nPlease directly answer the following question from the image description, without guessing or reasoning.\nQuestion: Is there a potted plant in the image?"
  },
  {
    "id": "train_coco_344408_cat_random_oven_handbag_no",
    "gt_answer": "no",
    "question": "Is there a handbag in the image?",
    "visual_prompt": "Question: Is there a handbag in the image?\nPlease answer the question based on the image.",
    "trusted_prompt_fact": "The given image depicts the following scene: There is one oven in the image. There is no handbag in the image.\nPlease directly answer the following question from the image description, without guessing or reasoning.\nQuestion: Is there a handbag in the image?",
    "trusted_prompt_counterfact": "The given image depicts the following scene: There is one oven in the image. There is a handbag in the image.\nPlease directly answer the following question from the image description, without guessing or reasoning.\nQuestion: Is there a handbag in the image?"
  }
]
```
### cat_popular
```json
[
  {
    "id": "val_coco_482978_cat_popular_chair_person_yes",
    "gt_answer": "yes",
    "question": "Is there a chair in the image?",
    "visual_prompt": "Question: Is there a chair in the image?\nPlease answer the question based on the image.",
    "trusted_prompt_fact": "The given image depicts the following scene: There is one train in the image. There is a chair in the image.\nPlease directly answer the following question from the image description, without guessing or reasoning.\nQuestion: Is there a chair in the image?",
    "trusted_prompt_counterfact": "The given image depicts the following scene: There is one train in the image. There is no chair in the image.\nPlease directly answer the following question from the image description, without guessing or reasoning.\nQuestion: Is there a chair in the image?"
  },
  {
    "id": "train_coco_549972_cat_popular_keyboard_person_no",
    "gt_answer": "no",
    "question": "Is there a person in the image?",
    "visual_prompt": "Question: Is there a person in the image?\nPlease answer the question based on the image.",
    "trusted_prompt_fact": "The given image depicts the following scene: There is one cup in the image. There is one keyboard in the image. There is no person in the image.\nPlease directly answer the following question from the image description, without guessing or reasoning.\nQuestion: Is there a person in the image?",
    "trusted_prompt_counterfact": "The given image depicts the following scene: There is one cup in the image. There is one keyboard in the image. There is a person in the image.\nPlease directly answer the following question from the image description, without guessing or reasoning.\nQuestion: Is there a person in the image?"
  },
  {
    "id": "train_coco_37548_cat_popular_elephant_person_no",
    "gt_answer": "no",
    "question": "Is there a person in the image?",
    "visual_prompt": "Question: Is there a person in the image?\nPlease answer the question based on the image.",
    "trusted_prompt_fact": "The given image depicts the following scene: There are three elephants in the image. There is no person in the image.\nPlease directly answer the following question from the image description, without guessing or reasoning.\nQuestion: Is there a person in the image?",
    "trusted_prompt_counterfact": "The given image depicts the following scene: There are three elephants in the image. There is a person in the image.\nPlease directly answer the following question from the image description, without guessing or reasoning.\nQuestion: Is there a person in the image?"
  },
  {
    "id": "train_coco_55849_cat_popular_baseball_bat_car_no",
    "gt_answer": "no",
    "question": "Is there a car in the image?",
    "visual_prompt": "Question: Is there a car in the image?\nPlease answer the question based on the image.",
    "trusted_prompt_fact": "The given image depicts the following scene: There are 7 persons in the image. There is one baseball bat in the image. There is no car in the image.\nPlease directly answer the following question from the image description, without guessing or reasoning.\nQuestion: Is there a car in the image?",
    "trusted_prompt_counterfact": "The given image depicts the following scene: There are 7 persons in the image. There is one baseball bat in the image. There is a car in the image.\nPlease directly answer the following question from the image description, without guessing or reasoning.\nQuestion: Is there a car in the image?"
  },
  {
    "id": "train_coco_127007_cat_popular_chair_person_yes",
    "gt_answer": "yes",
    "question": "Is there a chair in the image?",
    "visual_prompt": "Question: Is there a chair in the image?\nPlease answer the question based on the image.",
    "trusted_prompt_fact": "The given image depicts the following scene: There are four books in the image. There is one keyboard in the image. There is a chair in the image.\nPlease directly answer the following question from the image description, without guessing or reasoning.\nQuestion: Is there a chair in the image?",
    "trusted_prompt_counterfact": "The given image depicts the following scene: There are four books in the image. There is one keyboard in the image. There is no chair in the image.\nPlease directly answer the following question from the image description, without guessing or reasoning.\nQuestion: Is there a chair in the image?"
  }
]
```
### cat_hard
```json
[
  {
    "id": "train_coco_351622_cat_hard_person_cup_yes",
    "gt_answer": "yes",
    "question": "Is there a person in the image?",
    "visual_prompt": "Question: Is there a person in the image?\nPlease answer the question based on the image.",
    "trusted_prompt_fact": "The given image depicts the following scene: There are three cell phones in the image. There is one chair in the image. There is one dining table in the image. There is a person in the image.\nPlease directly answer the following question from the image description, without guessing or reasoning.\nQuestion: Is there a person in the image?",
    "trusted_prompt_counterfact": "The given image depicts the following scene: There are three cell phones in the image. There is one chair in the image. There is one dining table in the image. There is no person in the image.\nPlease directly answer the following question from the image description, without guessing or reasoning.\nQuestion: Is there a person in the image?"
  },
  {
    "id": "val_coco_398188_cat_hard_person_chair_yes",
    "gt_answer": "yes",
    "question": "Is there a person in the image?",
    "visual_prompt": "Question: Is there a person in the image?\nPlease answer the question based on the image.",
    "trusted_prompt_fact": "The given image depicts the following scene: There are two skateboards in the image. There is one bottle in the image. There is a person in the image.\nPlease directly answer the following question from the image description, without guessing or reasoning.\nQuestion: Is there a person in the image?",
    "trusted_prompt_counterfact": "The given image depicts the following scene: There are two skateboards in the image. There is one bottle in the image. There is no person in the image.\nPlease directly answer the following question from the image description, without guessing or reasoning.\nQuestion: Is there a person in the image?"
  },
  {
    "id": "train_coco_487114_cat_hard_banana_person_no",
    "gt_answer": "no",
    "question": "Is there a person in the image?",
    "visual_prompt": "Question: Is there a person in the image?\nPlease answer the question based on the image.",
    "trusted_prompt_fact": "The given image depicts the following scene: There are 7 bananas in the image. There is no person in the image.\nPlease directly answer the following question from the image description, without guessing or reasoning.\nQuestion: Is there a person in the image?",
    "trusted_prompt_counterfact": "The given image depicts the following scene: There are 7 bananas in the image. There is a person in the image.\nPlease directly answer the following question from the image description, without guessing or reasoning.\nQuestion: Is there a person in the image?"
  },
  {
    "id": "train_coco_299103_cat_hard_chair_person_no",
    "gt_answer": "no",
    "question": "Is there a person in the image?",
    "visual_prompt": "Question: Is there a person in the image?\nPlease answer the question based on the image.",
    "trusted_prompt_fact": "The given image depicts the following scene: There is one cat in the image. There is one chair in the image. There is one laptop in the image. There is one mouse in the image. There is no person in the image.\nPlease directly answer the following question from the image description, without guessing or reasoning.\nQuestion: Is there a person in the image?",
    "trusted_prompt_counterfact": "The given image depicts the following scene: There is one cat in the image. There is one chair in the image. There is one laptop in the image. There is one mouse in the image. There is a person in the image.\nPlease directly answer the following question from the image description, without guessing or reasoning.\nQuestion: Is there a person in the image?"
  },
  {
    "id": "train_coco_154680_cat_hard_fork_dining_table_yes",
    "gt_answer": "yes",
    "question": "Is there a fork in the image?",
    "visual_prompt": "Question: Is there a fork in the image?\nPlease answer the question based on the image.",
    "trusted_prompt_fact": "The given image depicts the following scene: There are two cups in the image. There is one sandwich in the image. There is a fork in the image.\nPlease directly answer the following question from the image description, without guessing or reasoning.\nQuestion: Is there a fork in the image?",
    "trusted_prompt_counterfact": "The given image depicts the following scene: There are two cups in the image. There is one sandwich in the image. There is no fork in the image.\nPlease directly answer the following question from the image description, without guessing or reasoning.\nQuestion: Is there a fork in the image?"
  }
]
```
### attr_color
```json
[
  {
    "id": "train_gqa_attr_color_2368476_2160825_gray_black_no",
    "gt_answer": "no",
    "question": "Is the post black?",
    "visual_prompt": "Question: Is the post black?\nPlease answer the question based on the image.",
    "trusted_prompt_fact": "The given image depicts the following scene: There is a post in the image. The post is gray.\nPlease directly answer the following question from the image description, without guessing or reasoning.\nQuestion: Is the post black?",
    "trusted_prompt_counterfact": "The given image depicts the following scene: There is a post in the image. The post is black.\nPlease directly answer the following question from the image description, without guessing or reasoning.\nQuestion: Is the post black?"
  },
  {
    "id": "train_gqa_attr_color_2400800_405301_black_white_yes",
    "gt_answer": "yes",
    "question": "Is the shoes black?",
    "visual_prompt": "Question: Is the shoes black?\nPlease answer the question based on the image.",
    "trusted_prompt_fact": "The given image depicts the following scene: There is a shoes in the image. The shoes is black.\nPlease directly answer the following question from the image description, without guessing or reasoning.\nQuestion: Is the shoes black?",
    "trusted_prompt_counterfact": "The given image depicts the following scene: There is a shoes in the image. The shoes is white.\nPlease directly answer the following question from the image description, without guessing or reasoning.\nQuestion: Is the shoes black?"
  },
  {
    "id": "train_gqa_attr_color_2357994_807073_blue_white_yes",
    "gt_answer": "yes",
    "question": "Is the leg blue?",
    "visual_prompt": "Question: Is the leg blue?\nPlease answer the question based on the image.",
    "trusted_prompt_fact": "The given image depicts the following scene: There is a leg in the image. The leg is blue.\nPlease directly answer the following question from the image description, without guessing or reasoning.\nQuestion: Is the leg blue?",
    "trusted_prompt_counterfact": "The given image depicts the following scene: There is a leg in the image. The leg is white.\nPlease directly answer the following question from the image description, without guessing or reasoning.\nQuestion: Is the leg blue?"
  },
  {
    "id": "train_gqa_attr_color_2346212_1899615_white_black_no",
    "gt_answer": "no",
    "question": "Is the hat black?",
    "visual_prompt": "Question: Is the hat black?\nPlease answer the question based on the image.",
    "trusted_prompt_fact": "The given image depicts the following scene: There is a hat in the image. The hat is white.\nPlease directly answer the following question from the image description, without guessing or reasoning.\nQuestion: Is the hat black?",
    "trusted_prompt_counterfact": "The given image depicts the following scene: There is a hat in the image. The hat is black.\nPlease directly answer the following question from the image description, without guessing or reasoning.\nQuestion: Is the hat black?"
  },
  {
    "id": "train_gqa_attr_color_2327649_3520176_green_brown_no",
    "gt_answer": "no",
    "question": "Is the leaf brown?",
    "visual_prompt": "Question: Is the leaf brown?\nPlease answer the question based on the image.",
    "trusted_prompt_fact": "The given image depicts the following scene: There is a leaf in the image. The leaf is green.\nPlease directly answer the following question from the image description, without guessing or reasoning.\nQuestion: Is the leaf brown?",
    "trusted_prompt_counterfact": "The given image depicts the following scene: There is a leaf in the image. The leaf is brown.\nPlease directly answer the following question from the image description, without guessing or reasoning.\nQuestion: Is the leaf brown?"
  }
]
```
### attr_count
```json
[
  {
    "id": "val_gqa_attr_count_2319039_watch_1_2_yes",
    "gt_answer": "yes",
    "question": "Are there one watch in the image?",
    "visual_prompt": "Question: Are there one watch in the image?\nPlease answer the question based on the image.",
    "trusted_prompt_fact": "The given image depicts the following scene: There is a watch in the image. There is one watch in the image.\nPlease directly answer the following question from the image description, without guessing or reasoning.\nQuestion: Are there one watch in the image?",
    "trusted_prompt_counterfact": "The given image depicts the following scene: There is a watch in the image. There are two watchs in the image.\nPlease directly answer the following question from the image description, without guessing or reasoning.\nQuestion: Are there one watch in the image?"
  },
  {
    "id": "val_gqa_attr_count_2330307_feet_1_2_no",
    "gt_answer": "no",
    "question": "Are there two feets in the image?",
    "visual_prompt": "Question: Are there two feets in the image?\nPlease answer the question based on the image.",
    "trusted_prompt_fact": "The given image depicts the following scene: There is a feet in the image. There is one feet in the image.\nPlease directly answer the following question from the image description, without guessing or reasoning.\nQuestion: Are there two feets in the image?",
    "trusted_prompt_counterfact": "The given image depicts the following scene: There is a feet in the image. There are two feets in the image.\nPlease directly answer the following question from the image description, without guessing or reasoning.\nQuestion: Are there two feets in the image?"
  },
  {
    "id": "val_gqa_attr_count_2367819_woman_1_2_yes",
    "gt_answer": "yes",
    "question": "Are there one woman in the image?",
    "visual_prompt": "Question: Are there one woman in the image?\nPlease answer the question based on the image.",
    "trusted_prompt_fact": "The given image depicts the following scene: There is a woman in the image. There is one woman in the image.\nPlease directly answer the following question from the image description, without guessing or reasoning.\nQuestion: Are there one woman in the image?",
    "trusted_prompt_counterfact": "The given image depicts the following scene: There is a woman in the image. There are two womans in the image.\nPlease directly answer the following question from the image description, without guessing or reasoning.\nQuestion: Are there one woman in the image?"
  },
  {
    "id": "val_gqa_attr_count_2414651_building_3_4_yes",
    "gt_answer": "yes",
    "question": "Are there three buildings in the image?",
    "visual_prompt": "Question: Are there three buildings in the image?\nPlease answer the question based on the image.",
    "trusted_prompt_fact": "The given image depicts the following scene: There is a building in the image. There are three buildings in the image.\nPlease directly answer the following question from the image description, without guessing or reasoning.\nQuestion: Are there three buildings in the image?",
    "trusted_prompt_counterfact": "The given image depicts the following scene: There is a building in the image. There are four buildings in the image.\nPlease directly answer the following question from the image description, without guessing or reasoning.\nQuestion: Are there three buildings in the image?"
  },
  {
    "id": "train_gqa_attr_count_2355507_wings_2_3_no",
    "gt_answer": "no",
    "question": "Are there three wings in the image?",
    "visual_prompt": "Question: Are there three wings in the image?\nPlease answer the question based on the image.",
    "trusted_prompt_fact": "The given image depicts the following scene: There is a wings in the image. There are two wings in the image.\nPlease directly answer the following question from the image description, without guessing or reasoning.\nQuestion: Are there three wings in the image?",
    "trusted_prompt_counterfact": "The given image depicts the following scene: There is a wings in the image. There are three wings in the image.\nPlease directly answer the following question from the image description, without guessing or reasoning.\nQuestion: Are there three wings in the image?"
  }
]
```
### rel_spatial
```json
[
  {
    "id": "train_gqa_rel_spatial_2368489_1899620_left_of_3868808_no",
    "gt_answer": "no",
    "question": "Is the plant right of the ground in the image?",
    "visual_prompt": "Question: Is the plant right of the ground in the image?\nPlease answer the question based on the image.",
    "trusted_prompt_fact": "The given image depicts the following scene: There are a ground, and a plant in the image. The plant is left of the ground in the image.\nPlease directly answer the following question from the image description, without guessing or reasoning.\nQuestion: Is the plant right of the ground in the image?",
    "trusted_prompt_counterfact": "The given image depicts the following scene: There are a ground, and a plant in the image. The plant is right of the ground in the image.\nPlease directly answer the following question from the image description, without guessing or reasoning.\nQuestion: Is the plant right of the ground in the image?"
  },
  {
    "id": "train_gqa_rel_spatial_2333416_4676572_left_of_4640525_yes",
    "gt_answer": "yes",
    "question": "Is the seafood left of the shrimp in the image?",
    "visual_prompt": "Question: Is the seafood left of the shrimp in the image?\nPlease answer the question based on the image.",
    "trusted_prompt_fact": "The given image depicts the following scene: There are a seafood, and a shrimp in the image. The seafood is left of the shrimp in the image.\nPlease directly answer the following question from the image description, without guessing or reasoning.\nQuestion: Is the seafood left of the shrimp in the image?",
    "trusted_prompt_counterfact": "The given image depicts the following scene: There are a seafood, and a shrimp in the image. The seafood is right of the shrimp in the image.\nPlease directly answer the following question from the image description, without guessing or reasoning.\nQuestion: Is the seafood left of the shrimp in the image?"
  },
  {
    "id": "train_gqa_rel_spatial_2368140_3467510_left_of_3601101_yes",
    "gt_answer": "yes",
    "question": "Is the bench left of the door in the image?",
    "visual_prompt": "Question: Is the bench left of the door in the image?\nPlease answer the question based on the image.",
    "trusted_prompt_fact": "The given image depicts the following scene: There are a bench, and a door in the image. The bench is left of the door in the image.\nPlease directly answer the following question from the image description, without guessing or reasoning.\nQuestion: Is the bench left of the door in the image?",
    "trusted_prompt_counterfact": "The given image depicts the following scene: There are a bench, and a door in the image. The bench is right of the door in the image.\nPlease directly answer the following question from the image description, without guessing or reasoning.\nQuestion: Is the bench left of the door in the image?"
  },
  {
    "id": "train_gqa_rel_spatial_2335980_2430699_right_of_2336924_yes",
    "gt_answer": "yes",
    "question": "Is the leaf right of the knife in the image?",
    "visual_prompt": "Question: Is the leaf right of the knife in the image?\nPlease answer the question based on the image.",
    "trusted_prompt_fact": "The given image depicts the following scene: There are a knife, and a leaf in the image. The leaf is right of the knife in the image.\nPlease directly answer the following question from the image description, without guessing or reasoning.\nQuestion: Is the leaf right of the knife in the image?",
    "trusted_prompt_counterfact": "The given image depicts the following scene: There are a knife, and a leaf in the image. The leaf is left of the knife in the image.\nPlease directly answer the following question from the image description, without guessing or reasoning.\nQuestion: Is the leaf right of the knife in the image?"
  },
  {
    "id": "train_gqa_rel_spatial_2406577_293415_right_of_293424_no",
    "gt_answer": "no",
    "question": "Is the beak left of the birds in the image?",
    "visual_prompt": "Question: Is the beak left of the birds in the image?\nPlease answer the question based on the image.",
    "trusted_prompt_fact": "The given image depicts the following scene: There are a beak, and a birds in the image. The beak is right of the birds in the image.\nPlease directly answer the following question from the image description, without guessing or reasoning.\nQuestion: Is the beak left of the birds in the image?",
    "trusted_prompt_counterfact": "The given image depicts the following scene: There are a beak, and a birds in the image. The beak is left of the birds in the image.\nPlease directly answer the following question from the image description, without guessing or reasoning.\nQuestion: Is the beak left of the birds in the image?"
  }
]
```
### rel_contact
```json
[
  {
    "id": "val_gqa_rel_contact_2352349_2529949_sitting_on_3783621_yes",
    "gt_answer": "yes",
    "question": "Is the man sitting on the ground in the image?",
    "visual_prompt": "Question: Is the man sitting on the ground in the image?\nPlease answer the question based on the image.",
    "trusted_prompt_fact": "The given image depicts the following scene: There are a ground, and a man in the image. The man is sitting on the ground in the image.\nPlease directly answer the following question from the image description, without guessing or reasoning.\nQuestion: Is the man sitting on the ground in the image?",
    "trusted_prompt_counterfact": "The given image depicts the following scene: There are a ground, and a man in the image. The man is standing next to the ground in the image.\nPlease directly answer the following question from the image description, without guessing or reasoning.\nQuestion: Is the man sitting on the ground in the image?"
  },
  {
    "id": "val_gqa_rel_contact_2340652_2004281_wearing_2004282_yes",
    "gt_answer": "yes",
    "question": "Is the boy wearing the shirt in the image?",
    "visual_prompt": "Question: Is the boy wearing the shirt in the image?\nPlease answer the question based on the image.",
    "trusted_prompt_fact": "The given image depicts the following scene: There are a boy, and a shirt in the image. The boy is wearing the shirt in the image.\nPlease directly answer the following question from the image description, without guessing or reasoning.\nQuestion: Is the boy wearing the shirt in the image?",
    "trusted_prompt_counterfact": "The given image depicts the following scene: There are a boy, and a shirt in the image. The boy is holding the shirt in the image.\nPlease directly answer the following question from the image description, without guessing or reasoning.\nQuestion: Is the boy wearing the shirt in the image?"
  },
  {
    "id": "train_gqa_rel_contact_2331317_3716596_wearing_3127436_yes",
    "gt_answer": "yes",
    "question": "Is the man wearing the jeans in the image?",
    "visual_prompt": "Question: Is the man wearing the jeans in the image?\nPlease answer the question based on the image.",
    "trusted_prompt_fact": "The given image depicts the following scene: There are a jeans, and a man in the image. The man is wearing the jeans in the image.\nPlease directly answer the following question from the image description, without guessing or reasoning.\nQuestion: Is the man wearing the jeans in the image?",
    "trusted_prompt_counterfact": "The given image depicts the following scene: There are a jeans, and a man in the image. The man is holding the jeans in the image.\nPlease directly answer the following question from the image description, without guessing or reasoning.\nQuestion: Is the man wearing the jeans in the image?"
  },
  {
    "id": "train_gqa_rel_contact_2415_1647917_carrying_1647918_no",
    "gt_answer": "no",
    "question": "Is the woman touching the bag in the image?",
    "visual_prompt": "Question: Is the woman touching the bag in the image?\nPlease answer the question based on the image.",
    "trusted_prompt_fact": "The given image depicts the following scene: There are a bag, and a woman in the image. The woman is carrying the bag in the image.\nPlease directly answer the following question from the image description, without guessing or reasoning.\nQuestion: Is the woman touching the bag in the image?",
    "trusted_prompt_counterfact": "The given image depicts the following scene: There are a bag, and a woman in the image. The woman is touching the bag in the image.\nPlease directly answer the following question from the image description, without guessing or reasoning.\nQuestion: Is the woman touching the bag in the image?"
  },
  {
    "id": "train_gqa_rel_contact_2413388_174555_holding_174557_yes",
    "gt_answer": "yes",
    "question": "Is the woman holding the umbrella in the image?",
    "visual_prompt": "Question: Is the woman holding the umbrella in the image?\nPlease answer the question based on the image.",
    "trusted_prompt_fact": "The given image depicts the following scene: There are an umbrella, and a woman in the image. The woman is holding the umbrella in the image.\nPlease directly answer the following question from the image description, without guessing or reasoning.\nQuestion: Is the woman holding the umbrella in the image?",
    "trusted_prompt_counterfact": "The given image depicts the following scene: There are an umbrella, and a woman in the image. The woman is wearing the umbrella in the image.\nPlease directly answer the following question from the image description, without guessing or reasoning.\nQuestion: Is the woman holding the umbrella in the image?"
  }
]
```


## 3. Activation Extraction Summary
```json
{
  "script": "scripts/merge_subtype_minpair_activations.py",
  "model_path": "/home/huiwei/sy/models/llava-v1.5-7b-official-clean",
  "model_name": "llava-v1.5-7b-official-clean",
  "context_len": 2048,
  "llava_repo_path": "/home/huiwei/sy/LLaVA-official-clean",
  "conv_mode": "llava_v1",
  "storage_dtype": "float16",
  "num_layers": 32,
  "num_heads": 32,
  "head_dim": 128,
  "shape": [
    3800,
    32,
    32,
    128
  ],
  "counts_by_subtype": {
    "attr_color": 125,
    "attr_count": 125,
    "cat_hard": 150,
    "cat_popular": 150,
    "cat_random": 150,
    "rel_contact": 125,
    "rel_spatial": 125
  },
  "num_shards": 4,
  "shard_index": 0,
  "source_jsonl": "/home/huiwei/sy/halludata/data/subtype_minpair_v1/minimal_pairs/train.jsonl",
  "branch_definitions": {
    "z_visual": "image + visual_prompt",
    "z_fact_text": "trusted_prompt_fact, text-only",
    "z_counterfact_text": "trusted_prompt_counterfact, text-only"
  },
  "num_shards_merged": 4,
  "shard_files": [
    "/home/huiwei/sy/halludata/data/subtype_minpair_v1/activations/train_shard0.pt",
    "/home/huiwei/sy/halludata/data/subtype_minpair_v1/activations/train_shard1.pt",
    "/home/huiwei/sy/halludata/data/subtype_minpair_v1/activations/train_shard2.pt",
    "/home/huiwei/sy/halludata/data/subtype_minpair_v1/activations/train_shard3.pt"
  ],
  "metadata_output": "/home/huiwei/sy/halludata/data/subtype_minpair_v1/activations/train_activations.meta.jsonl",
  "yesno_output": "/home/huiwei/sy/halludata/data/subtype_minpair_v1/activations/train_activations.yesno.pt"
}
```

## 4. Vector Construction Summary
# Subtype Minimal-Pair Vector Report

- Source activations: `/home/huiwei/sy/halludata/data/subtype_minpair_v1/minimal_pairs/train.jsonl`
- Vector shape: `[32, 32, 128]`
- Samples: `3800`
- Yes/no direction mode: `answer_token`

## Counts
| subtype | count |
| --- | --- |
| attr_color | 500 |
| attr_count | 500 |
| cat_hard | 600 |
| cat_popular | 600 |
| cat_random | 600 |
| rel_contact | 500 |
| rel_spatial | 500 |

## Denoising
| vector | num_samples | method | svd_k | first_singular | top_energy | mean_only_norm | projected_norm |
| --- | --- | --- | --- | --- | --- | --- | --- |
| g_all | 3800 | uncentered_svd | 4 | 46.9193 | 0.5793 | 0.7595 | 0.7595 |
| g_cat | 1800 | uncentered_svd | 4 | 34.9479 | 0.6785 | 0.8231 | 0.8231 |
| g_attr | 1000 | uncentered_svd | 4 | 24.2512 | 0.5881 | 0.7662 | 0.7662 |
| g_rel | 1000 | uncentered_svd | 4 | 25.0162 | 0.6258 | 0.7903 | 0.7903 |
| s_cat_random | 600 | uncentered_svd | 4 | 22.3606 | 0.8333 | 0.1129 | 0.1102 |
| s_cat_popular | 600 | uncentered_svd | 4 | 22.3272 | 0.8308 | 0.1259 | 0.1195 |
| s_cat_hard | 600 | uncentered_svd | 4 | 22.2911 | 0.8282 | 0.1027 | 0.0979 |
| s_attr_color | 500 | uncentered_svd | 4 | 20.3620 | 0.8292 | 0.0452 | 0.0386 |
| s_attr_count | 500 | uncentered_svd | 4 | 18.1026 | 0.6554 | 0.4129 | 0.4128 |
| s_rel_spatial | 500 | uncentered_svd | 4 | 18.7387 | 0.7023 | 0.0240 | 0.0068 |
| s_rel_contact | 500 | uncentered_svd | 4 | 19.1405 | 0.7327 | 0.2195 | 0.2168 |

## Yes/No Projection
| vector | raw_yesno_cosine | clean_yesno_cosine | projection_norm_ratio | clean_norm_over_raw_norm |
| --- | --- | --- | --- | --- |
| g_all | -0.0065 | 0.0000 | 0.0065 | 1.0000 |
| g_cat | -0.0068 | 0.0000 | 0.0068 | 1.0000 |
| g_attr | -0.0122 | 0.0000 | 0.0122 | 0.9999 |
| g_rel | 0.0009 | 0.0000 | 0.0009 | 1.0000 |
| s_cat_random | 0.0302 | -0.0000 | 0.0302 | 0.9995 |
| s_cat_popular | 0.0278 | -0.0000 | 0.0278 | 0.9996 |
| s_cat_hard | 0.0309 | -0.0000 | 0.0309 | 0.9995 |
| s_attr_color | 0.0093 | -0.0000 | 0.0093 | 1.0000 |
| s_attr_count | -0.0170 | 0.0000 | 0.0170 | 0.9999 |
| s_rel_spatial | 0.0053 | -0.0000 | 0.0053 | 1.0000 |
| s_rel_contact | -0.0045 | -0.0000 | 0.0045 | 1.0000 |

## Clean Cosine Matrix
| vector | g_all_clean | g_cat_clean | g_attr_clean | g_rel_clean | s_cat_hard_clean | s_attr_color_clean | s_attr_count_clean | s_rel_spatial_clean | s_rel_contact_clean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| g_all_clean | 1.0000 | 0.9634 | 0.9408 | 0.9333 | 0.0041 | -0.0633 | 0.0783 | -0.0156 | 0.0537 |
| g_cat_clean | 0.9634 | 1.0000 | 0.8435 | 0.8255 | -0.0360 | -0.0840 | 0.0491 | -0.0030 | 0.0405 |
| g_attr_clean | 0.9408 | 0.8435 | 1.0000 | 0.8857 | 0.0515 | -0.0570 | 0.1505 | -0.0392 | 0.0480 |
| g_rel_clean | 0.9333 | 0.8255 | 0.8857 | 1.0000 | 0.0309 | -0.0186 | 0.0464 | -0.0125 | 0.0722 |
| s_cat_hard_clean | 0.0041 | -0.0360 | 0.0515 | 0.0309 | 1.0000 | 0.0168 | 0.1102 | 0.0155 | -0.0001 |
| s_attr_color_clean | -0.0633 | -0.0840 | -0.0570 | -0.0186 | 0.0168 | 1.0000 | -0.0494 | -0.0485 | 0.1759 |
| s_attr_count_clean | 0.0783 | 0.0491 | 0.1505 | 0.0464 | 0.1102 | -0.0494 | 1.0000 | -0.0270 | 0.0546 |
| s_rel_spatial_clean | -0.0156 | -0.0030 | -0.0392 | -0.0125 | 0.0155 | -0.0485 | -0.0270 | 1.0000 | 0.0261 |
| s_rel_contact_clean | 0.0537 | 0.0405 | 0.0480 | 0.0722 | -0.0001 | 0.1759 | 0.0546 | 0.0261 | 1.0000 |

## Vector Norms
| vector | flat_norm | head_norm_mean | head_norm_max | finite |
| --- | --- | --- | --- | --- |
| d_attr_color_g1_s025_clean | 0.4023 | 0.0087 | 0.0683 | True |
| d_attr_color_g1_s025_unit_clean | 1.0000 | 0.0216 | 0.1697 | True |
| d_attr_color_g1_s05_clean | 0.4023 | 0.0086 | 0.0654 | True |
| d_attr_color_g1_s05_unit_clean | 1.0000 | 0.0213 | 0.1624 | True |
| d_attr_color_g1_s1_clean | 0.4023 | 0.0081 | 0.0990 | True |
| d_attr_color_g1_s1_unit_clean | 1.0000 | 0.0201 | 0.2461 | True |
| d_attr_color_g_only_clean | 0.4023 | 0.0087 | 0.0744 | True |
| d_attr_color_g_only_unit_clean | 1.0000 | 0.0216 | 0.1850 | True |
| d_attr_color_s_only_clean | 0.4023 | 0.0064 | 0.1333 | True |
| d_attr_color_s_only_unit_clean | 1.0000 | 0.0158 | 0.3313 | True |
| d_attr_count_g1_s025_clean | 0.5894 | 0.0125 | 0.1188 | True |
| d_attr_count_g1_s025_unit_clean | 1.0000 | 0.0213 | 0.2016 | True |
| d_attr_count_g1_s05_clean | 0.5894 | 0.0123 | 0.1246 | True |
| d_attr_count_g1_s05_unit_clean | 1.0000 | 0.0208 | 0.2113 | True |
| d_attr_count_g1_s1_clean | 0.5894 | 0.0116 | 0.1278 | True |
| d_attr_count_g1_s1_unit_clean | 1.0000 | 0.0197 | 0.2168 | True |
| d_attr_count_g_only_clean | 0.5894 | 0.0127 | 0.1091 | True |
| d_attr_count_g_only_unit_clean | 1.0000 | 0.0216 | 0.1850 | True |
| d_attr_count_s_only_clean | 0.5894 | 0.0096 | 0.1541 | True |
| d_attr_count_s_only_unit_clean | 1.0000 | 0.0163 | 0.2614 | True |
| d_cat_hard_g1_s025_clean | 0.4604 | 0.0099 | 0.0683 | True |
| d_cat_hard_g1_s025_unit_clean | 1.0000 | 0.0214 | 0.1484 | True |
| d_cat_hard_g1_s05_clean | 0.4604 | 0.0097 | 0.0691 | True |
| d_cat_hard_g1_s05_unit_clean | 1.0000 | 0.0211 | 0.1500 | True |
| d_cat_hard_g1_s1_clean | 0.4604 | 0.0092 | 0.0757 | True |
| d_cat_hard_g1_s1_unit_clean | 1.0000 | 0.0201 | 0.1644 | True |
| d_cat_hard_g_only_clean | 0.4604 | 0.0099 | 0.0661 | True |
| d_cat_hard_g_only_unit_clean | 1.0000 | 0.0215 | 0.1436 | True |
| d_cat_hard_s_only_clean | 0.4604 | 0.0075 | 0.0982 | True |
| d_cat_hard_s_only_unit_clean | 1.0000 | 0.0163 | 0.2132 | True |
| d_cat_popular_g1_s025_clean | 0.4713 | 0.0101 | 0.0701 | True |
| d_cat_popular_g1_s025_unit_clean | 1.0000 | 0.0214 | 0.1487 | True |
| d_cat_popular_g1_s05_clean | 0.4713 | 0.0100 | 0.0711 | True |
| d_cat_popular_g1_s05_unit_clean | 1.0000 | 0.0211 | 0.1508 | True |
| d_cat_popular_g1_s1_clean | 0.4713 | 0.0095 | 0.0790 | True |
| d_cat_popular_g1_s1_unit_clean | 1.0000 | 0.0201 | 0.1677 | True |
| d_cat_popular_g_only_clean | 0.4713 | 0.0101 | 0.0677 | True |
| d_cat_popular_g_only_unit_clean | 1.0000 | 0.0215 | 0.1436 | True |
| d_cat_popular_s_only_clean | 0.4713 | 0.0077 | 0.1020 | True |
| d_cat_popular_s_only_unit_clean | 1.0000 | 0.0162 | 0.2164 | True |
| d_cat_random_g1_s025_clean | 0.4666 | 0.0100 | 0.0640 | True |
| d_cat_random_g1_s025_unit_clean | 1.0000 | 0.0214 | 0.1372 | True |
| d_cat_random_g1_s05_clean | 0.4666 | 0.0098 | 0.0772 | True |
| d_cat_random_g1_s05_unit_clean | 1.0000 | 0.0210 | 0.1655 | True |
| d_cat_random_g1_s1_clean | 0.4666 | 0.0093 | 0.0994 | True |
| d_cat_random_g1_s1_unit_clean | 1.0000 | 0.0199 | 0.2131 | True |
| d_cat_random_g_only_clean | 0.4666 | 0.0100 | 0.0670 | True |
| d_cat_random_g_only_unit_clean | 1.0000 | 0.0215 | 0.1436 | True |
| d_cat_random_s_only_clean | 0.4666 | 0.0077 | 0.1185 | True |
| d_cat_random_s_only_unit_clean | 1.0000 | 0.0165 | 0.2540 | True |
| d_rel_contact_g1_s025_clean | 0.5035 | 0.0108 | 0.0854 | True |
| d_rel_contact_g1_s025_unit_clean | 1.0000 | 0.0215 | 0.1696 | True |
| d_rel_contact_g1_s05_clean | 0.5035 | 0.0106 | 0.0808 | True |
| d_rel_contact_g1_s05_unit_clean | 1.0000 | 0.0211 | 0.1604 | True |
| d_rel_contact_g1_s1_clean | 0.5035 | 0.0100 | 0.0855 | True |
| d_rel_contact_g1_s1_unit_clean | 1.0000 | 0.0199 | 0.1697 | True |
| d_rel_contact_g_only_clean | 0.5035 | 0.0108 | 0.0862 | True |
| d_rel_contact_g_only_unit_clean | 1.0000 | 0.0215 | 0.1712 | True |
| d_rel_contact_s_only_clean | 0.5035 | 0.0079 | 0.1114 | True |
| d_rel_contact_s_only_unit_clean | 1.0000 | 0.0157 | 0.2212 | True |
| d_rel_spatial_g1_s025_clean | 0.3985 | 0.0085 | 0.0670 | True |
| d_rel_spatial_g1_s025_unit_clean | 1.0000 | 0.0214 | 0.1680 | True |
| d_rel_spatial_g1_s05_clean | 0.3985 | 0.0083 | 0.0729 | True |
| d_rel_spatial_g1_s05_unit_clean | 1.0000 | 0.0209 | 0.1830 | True |
| d_rel_spatial_g1_s1_clean | 0.3985 | 0.0077 | 0.1121 | True |
| d_rel_spatial_g1_s1_unit_clean | 1.0000 | 0.0193 | 0.2812 | True |
| d_rel_spatial_g_only_clean | 0.3985 | 0.0086 | 0.0682 | True |
| d_rel_spatial_g_only_unit_clean | 1.0000 | 0.0215 | 0.1712 | True |
| d_rel_spatial_s_only_clean | 0.3985 | 0.0058 | 0.1582 | True |
| d_rel_spatial_s_only_unit_clean | 1.0000 | 0.0145 | 0.3968 | True |
| g_all_clean | 0.7594 | 0.0165 | 0.1239 | True |
| g_all_mean_only | 0.7595 | 0.0165 | 0.1240 | True |
| g_all_raw | 0.7595 | 0.0165 | 0.1239 | True |
| g_attr_clean | 0.7661 | 0.0165 | 0.1417 | True |
| g_attr_mean_only | 0.7662 | 0.0165 | 0.1417 | True |
| g_attr_raw | 0.7662 | 0.0165 | 0.1418 | True |
| g_cat_clean | 0.8231 | 0.0177 | 0.1182 | True |
| g_cat_mean_only | 0.8231 | 0.0177 | 0.1181 | True |
| g_cat_raw | 0.8231 | 0.0177 | 0.1182 | True |
| g_rel_clean | 0.7903 | 0.0170 | 0.1353 | True |
| g_rel_mean_only | 0.7903 | 0.0170 | 0.1353 | True |
| g_rel_raw | 0.7903 | 0.0170 | 0.1353 | True |
| s_attr_color_clean | 0.0386 | 0.0006 | 0.0128 | True |
| s_attr_color_mean_only | 0.0452 | 0.0007 | 0.0168 | True |
| s_attr_color_raw | 0.0386 | 0.0006 | 0.0128 | True |
| s_attr_count_clean | 0.4128 | 0.0067 | 0.1079 | True |
| s_attr_count_mean_only | 0.4129 | 0.0067 | 0.1075 | True |
| s_attr_count_raw | 0.4128 | 0.0067 | 0.1080 | True |
| s_cat_hard_clean | 0.0978 | 0.0016 | 0.0209 | True |
| s_cat_hard_mean_only | 0.1027 | 0.0016 | 0.0237 | True |
| s_cat_hard_raw | 0.0979 | 0.0016 | 0.0209 | True |
| s_cat_popular_clean | 0.1195 | 0.0019 | 0.0258 | True |
| s_cat_popular_mean_only | 0.1259 | 0.0020 | 0.0278 | True |
| s_cat_popular_raw | 0.1195 | 0.0019 | 0.0259 | True |
| s_cat_random_clean | 0.1102 | 0.0018 | 0.0280 | True |
| s_cat_random_mean_only | 0.1129 | 0.0018 | 0.0309 | True |
| s_cat_random_raw | 0.1102 | 0.0018 | 0.0280 | True |
| s_rel_contact_clean | 0.2168 | 0.0034 | 0.0479 | True |
| s_rel_contact_mean_only | 0.2195 | 0.0034 | 0.0473 | True |
| s_rel_contact_raw | 0.2168 | 0.0034 | 0.0479 | True |
| s_rel_spatial_clean | 0.0068 | 0.0001 | 0.0027 | True |
| s_rel_spatial_mean_only | 0.0240 | 0.0004 | 0.0081 | True |
| s_rel_spatial_raw | 0.0068 | 0.0001 | 0.0027 | True |
| yesno_direction | 10.7614 | 0.2138 | 2.4315 | True |

## Top64 Head Overlap
| vector_a | vector_b | intersection | jaccard |
| --- | --- | --- | --- |
| g_all_clean | g_cat_clean | 58 | 0.8286 |
| g_all_clean | g_attr_clean | 58 | 0.8286 |
| g_all_clean | g_rel_clean | 58 | 0.8286 |
| g_all_clean | s_cat_hard_clean | 28 | 0.2800 |
| g_all_clean | s_attr_color_clean | 26 | 0.2549 |
| g_all_clean | s_attr_count_clean | 32 | 0.3333 |
| g_all_clean | s_rel_spatial_clean | 32 | 0.3333 |
| g_all_clean | s_rel_contact_clean | 27 | 0.2673 |
| g_cat_clean | g_attr_clean | 52 | 0.6842 |
| g_cat_clean | g_rel_clean | 54 | 0.7297 |
| g_cat_clean | s_cat_hard_clean | 27 | 0.2673 |
| g_cat_clean | s_attr_color_clean | 26 | 0.2549 |
| g_cat_clean | s_attr_count_clean | 33 | 0.3474 |
| g_cat_clean | s_rel_spatial_clean | 31 | 0.3196 |
| g_cat_clean | s_rel_contact_clean | 26 | 0.2549 |
| g_attr_clean | g_rel_clean | 57 | 0.8028 |
| g_attr_clean | s_cat_hard_clean | 28 | 0.2800 |
| g_attr_clean | s_attr_color_clean | 26 | 0.2549 |
| g_attr_clean | s_attr_count_clean | 34 | 0.3617 |
| g_attr_clean | s_rel_spatial_clean | 34 | 0.3617 |
| g_attr_clean | s_rel_contact_clean | 30 | 0.3061 |
| g_rel_clean | s_cat_hard_clean | 30 | 0.3061 |
| g_rel_clean | s_attr_color_clean | 28 | 0.2800 |
| g_rel_clean | s_attr_count_clean | 32 | 0.3333 |
| g_rel_clean | s_rel_spatial_clean | 31 | 0.3196 |
| g_rel_clean | s_rel_contact_clean | 29 | 0.2929 |
| s_cat_hard_clean | s_attr_color_clean | 39 | 0.4382 |
| s_cat_hard_clean | s_attr_count_clean | 47 | 0.5802 |
| s_cat_hard_clean | s_rel_spatial_clean | 39 | 0.4382 |
| s_cat_hard_clean | s_rel_contact_clean | 44 | 0.5238 |
| s_attr_color_clean | s_attr_count_clean | 40 | 0.4545 |
| s_attr_color_clean | s_rel_spatial_clean | 40 | 0.4545 |
| s_attr_color_clean | s_rel_contact_clean | 37 | 0.4066 |
| s_attr_count_clean | s_rel_spatial_clean | 43 | 0.5059 |
| s_attr_count_clean | s_rel_contact_clean | 49 | 0.6203 |
| s_rel_spatial_clean | s_rel_contact_clean | 42 | 0.4884 |

### g_all_clean Top64 Heads
| layer | head | norm |
| --- | --- | --- |
| 17 | 5 | 0.1239 |
| 15 | 17 | 0.1014 |
| 12 | 11 | 0.0999 |
| 31 | 22 | 0.0922 |
| 31 | 4 | 0.0909 |
| 13 | 2 | 0.0900 |
| 16 | 0 | 0.0888 |
| 30 | 31 | 0.0875 |
| 13 | 4 | 0.0872 |
| 11 | 8 | 0.0833 |
| 12 | 30 | 0.0804 |
| 12 | 1 | 0.0801 |
| 12 | 19 | 0.0740 |
| 14 | 20 | 0.0733 |
| 10 | 23 | 0.0710 |
| 24 | 23 | 0.0691 |
| 14 | 0 | 0.0679 |
| 9 | 21 | 0.0676 |
| 14 | 7 | 0.0675 |
| 13 | 1 | 0.0672 |

### g_cat_clean Top64 Heads
| layer | head | norm |
| --- | --- | --- |
| 17 | 5 | 0.1182 |
| 13 | 4 | 0.1030 |
| 15 | 17 | 0.1024 |
| 12 | 11 | 0.1020 |
| 12 | 1 | 0.1013 |
| 13 | 2 | 0.0998 |
| 31 | 4 | 0.0963 |
| 14 | 20 | 0.0950 |
| 16 | 0 | 0.0948 |
| 11 | 8 | 0.0932 |
| 30 | 31 | 0.0913 |
| 13 | 16 | 0.0885 |
| 15 | 31 | 0.0846 |
| 12 | 30 | 0.0835 |
| 12 | 19 | 0.0829 |
| 14 | 27 | 0.0778 |
| 14 | 0 | 0.0753 |
| 31 | 22 | 0.0750 |
| 16 | 14 | 0.0749 |
| 16 | 15 | 0.0727 |

### g_attr_clean Top64 Heads
| layer | head | norm |
| --- | --- | --- |
| 17 | 5 | 0.1417 |
| 15 | 17 | 0.1033 |
| 16 | 0 | 0.0973 |
| 13 | 4 | 0.0923 |
| 12 | 11 | 0.0905 |
| 30 | 31 | 0.0893 |
| 14 | 7 | 0.0835 |
| 31 | 4 | 0.0819 |
| 31 | 22 | 0.0804 |
| 13 | 2 | 0.0780 |
| 12 | 1 | 0.0778 |
| 14 | 0 | 0.0748 |
| 11 | 8 | 0.0743 |
| 22 | 17 | 0.0717 |
| 14 | 20 | 0.0713 |
| 9 | 21 | 0.0696 |
| 16 | 2 | 0.0692 |
| 14 | 24 | 0.0692 |
| 12 | 19 | 0.0689 |
| 9 | 20 | 0.0683 |

### g_rel_clean Top64 Heads
| layer | head | norm |
| --- | --- | --- |
| 31 | 22 | 0.1353 |
| 17 | 5 | 0.1308 |
| 12 | 11 | 0.1064 |
| 15 | 17 | 0.1010 |
| 12 | 30 | 0.0931 |
| 16 | 0 | 0.0930 |
| 31 | 4 | 0.0926 |
| 13 | 2 | 0.0910 |
| 13 | 4 | 0.0874 |
| 30 | 31 | 0.0870 |
| 11 | 8 | 0.0793 |
| 14 | 7 | 0.0773 |
| 12 | 1 | 0.0761 |
| 10 | 23 | 0.0758 |
| 24 | 23 | 0.0739 |
| 22 | 17 | 0.0737 |
| 16 | 2 | 0.0734 |
| 11 | 30 | 0.0729 |
| 12 | 19 | 0.0706 |
| 27 | 7 | 0.0697 |

### s_cat_hard_clean Top64 Heads
| layer | head | norm |
| --- | --- | --- |
| 14 | 27 | 0.0209 |
| 12 | 12 | 0.0194 |
| 14 | 3 | 0.0175 |
| 13 | 4 | 0.0165 |
| 12 | 1 | 0.0162 |
| 12 | 0 | 0.0162 |
| 14 | 23 | 0.0153 |
| 16 | 30 | 0.0151 |
| 13 | 30 | 0.0146 |
| 13 | 16 | 0.0146 |
| 15 | 31 | 0.0139 |
| 15 | 5 | 0.0131 |
| 13 | 22 | 0.0127 |
| 12 | 25 | 0.0127 |
| 14 | 4 | 0.0126 |
| 16 | 31 | 0.0126 |
| 16 | 17 | 0.0123 |
| 12 | 3 | 0.0123 |
| 17 | 5 | 0.0120 |
| 16 | 15 | 0.0118 |

### s_attr_color_clean Top64 Heads
| layer | head | norm |
| --- | --- | --- |
| 16 | 18 | 0.0128 |
| 14 | 27 | 0.0112 |
| 17 | 5 | 0.0075 |
| 16 | 0 | 0.0062 |
| 12 | 1 | 0.0061 |
| 13 | 4 | 0.0059 |
| 14 | 20 | 0.0057 |
| 16 | 30 | 0.0057 |
| 17 | 8 | 0.0055 |
| 15 | 5 | 0.0053 |
| 16 | 15 | 0.0052 |
| 13 | 30 | 0.0050 |
| 16 | 13 | 0.0049 |
| 11 | 3 | 0.0046 |
| 13 | 16 | 0.0044 |
| 15 | 31 | 0.0044 |
| 26 | 14 | 0.0042 |
| 21 | 28 | 0.0040 |
| 15 | 18 | 0.0039 |
| 23 | 31 | 0.0039 |

### s_attr_count_clean Top64 Heads
| layer | head | norm |
| --- | --- | --- |
| 14 | 27 | 0.1079 |
| 14 | 28 | 0.0801 |
| 16 | 18 | 0.0780 |
| 16 | 0 | 0.0771 |
| 17 | 5 | 0.0767 |
| 12 | 4 | 0.0731 |
| 12 | 1 | 0.0648 |
| 14 | 19 | 0.0612 |
| 15 | 31 | 0.0602 |
| 16 | 15 | 0.0577 |
| 13 | 4 | 0.0575 |
| 13 | 30 | 0.0548 |
| 15 | 5 | 0.0536 |
| 15 | 18 | 0.0494 |
| 16 | 30 | 0.0458 |
| 16 | 13 | 0.0448 |
| 14 | 3 | 0.0447 |
| 11 | 3 | 0.0446 |
| 12 | 0 | 0.0431 |
| 14 | 24 | 0.0431 |

### s_rel_spatial_clean Top64 Heads
| layer | head | norm |
| --- | --- | --- |
| 14 | 27 | 0.0027 |
| 16 | 18 | 0.0018 |
| 15 | 5 | 0.0016 |
| 12 | 1 | 0.0012 |
| 16 | 15 | 0.0012 |
| 16 | 0 | 0.0012 |
| 16 | 13 | 0.0011 |
| 13 | 4 | 0.0011 |
| 17 | 5 | 0.0011 |
| 18 | 26 | 0.0011 |
| 15 | 31 | 0.0010 |
| 13 | 30 | 0.0010 |
| 17 | 11 | 0.0010 |
| 16 | 30 | 0.0008 |
| 15 | 20 | 0.0008 |
| 13 | 22 | 0.0007 |
| 13 | 16 | 0.0007 |
| 16 | 2 | 0.0007 |
| 17 | 8 | 0.0007 |
| 14 | 2 | 0.0007 |

### s_rel_contact_clean Top64 Heads
| layer | head | norm |
| --- | --- | --- |
| 15 | 5 | 0.0479 |
| 16 | 18 | 0.0451 |
| 14 | 27 | 0.0439 |
| 17 | 5 | 0.0412 |
| 16 | 0 | 0.0404 |
| 15 | 31 | 0.0397 |
| 12 | 1 | 0.0374 |
| 16 | 13 | 0.0373 |
| 16 | 15 | 0.0371 |
| 13 | 4 | 0.0342 |
| 15 | 19 | 0.0338 |
| 16 | 3 | 0.0329 |
| 14 | 20 | 0.0310 |
| 12 | 4 | 0.0300 |
| 24 | 4 | 0.0300 |
| 12 | 25 | 0.0291 |
| 12 | 10 | 0.0244 |
| 13 | 16 | 0.0238 |
| 13 | 30 | 0.0237 |
| 18 | 26 | 0.0227 |

## Automatic Interpretation
- Yes/no projection removal succeeded numerically; clean vectors are nearly orthogonal to the yes/no direction.


## 5. Evaluation Summary
### Held-Out Best Rows
| eval_subset | vector | match | alpha | accuracy | f1 | yes_rate | wrong_to_right | right_to_wrong |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| attr_color | g_attr_clean | mismatched | 0.25 | 0.7300 | 0.7245 | 0.4800 | 30.0000 | 18.0000 |
| attr_color | g_cat_clean | mismatched | 0.5 | 0.7150 | 0.7136 | 0.4950 | 38.0000 | 29.0000 |
| attr_color | g_rel_clean | mismatched | 0.05 | 0.7100 | 0.7129 | 0.5100 | 39.0000 | 31.0000 |
| attr_color | d_cat_hard_g1_s05_clean | mismatched | 0.5 | 0.7000 | 0.7059 | 0.5200 | 34.0000 | 28.0000 |
| attr_color | d_attr_count_g1_s05_clean | mismatched | 0.25 | 0.6850 | 0.6897 | 0.5150 | 33.0000 | 30.0000 |
| attr_color | d_rel_spatial_g1_s05_clean | mismatched | 0.25 | 0.6750 | 0.6829 | 0.5250 | 34.0000 | 33.0000 |
| attr_color | d_rel_contact_g1_s05_clean | mismatched | 0.1 | 0.6700 | 0.6733 | 0.5100 | 30.0000 | 30.0000 |
| attr_color | g_all_clean | global | 0.1 | 0.6450 | 0.6667 | 0.5650 | 27.0000 | 32.0000 |
| attr_color | d_attr_color_g1_s05_clean | matched | 0.1 | 0.6600 | 0.6667 | 0.5200 | 32.0000 | 34.0000 |
| attr_count | d_rel_spatial_g1_s05_clean | mismatched | 0.1 | 0.6050 | 0.6030 | 0.4950 | 48.0000 | 36.0000 |
| attr_count | g_attr_clean | mismatched | 0.1 | 0.6050 | 0.5990 | 0.4850 | 54.0000 | 42.0000 |
| attr_count | d_attr_count_g1_s05_clean | matched | 0.1 | 0.5750 | 0.5933 | 0.5450 | 45.0000 | 39.0000 |
| attr_count | d_rel_contact_g1_s05_clean | mismatched | 0.25 | 0.5750 | 0.5729 | 0.4950 | 44.0000 | 38.0000 |
| attr_count | g_all_clean | global | 0.25 | 0.5350 | 0.5674 | 0.5750 | 45.0000 | 47.0000 |
| attr_count | g_cat_clean | mismatched | 0.05 | 0.5800 | 0.5670 | 0.4700 | 43.0000 | 36.0000 |
| attr_count | g_rel_clean | mismatched | 0.25 | 0.5500 | 0.5500 | 0.5000 | 42.0000 | 41.0000 |
| attr_count | d_cat_hard_g1_s05_clean | mismatched | 0.5 | 0.5500 | 0.5455 | 0.4900 | 44.0000 | 43.0000 |
| attr_count | d_attr_color_g1_s05_clean | mismatched | 0.05 | 0.5400 | 0.5306 | 0.4800 | 46.0000 | 47.0000 |
| cat_hard | d_attr_color_g1_s05_clean | mismatched | 0.5 | 0.9250 | 0.9231 | 0.4750 | 13.0000 | 3.0000 |
| cat_hard | g_attr_clean | mismatched | 0.25 | 0.8950 | 0.8945 | 0.4950 | 11.0000 | 7.0000 |
| cat_hard | d_rel_contact_g1_s05_clean | mismatched | 0.1 | 0.8900 | 0.8889 | 0.4900 | 14.0000 | 11.0000 |
| cat_hard | d_attr_count_g1_s05_clean | mismatched | 0.05 | 0.8900 | 0.8854 | 0.4600 | 11.0000 | 8.0000 |
| cat_hard | g_rel_clean | mismatched | 0.25 | 0.8900 | 0.8842 | 0.4500 | 12.0000 | 9.0000 |
| cat_hard | d_rel_spatial_g1_s05_clean | mismatched | 0.25 | 0.8850 | 0.8821 | 0.4750 | 14.0000 | 12.0000 |
| cat_hard | d_cat_hard_g1_s05_clean | matched | 0.1 | 0.8800 | 0.8763 | 0.4700 | 9.0000 | 8.0000 |
| cat_hard | g_cat_clean | matched | 0.1 | 0.8750 | 0.8744 | 0.4950 | 12.0000 | 12.0000 |
| cat_hard | g_all_clean | global | 0.1 | 0.8750 | 0.8718 | 0.4750 | 11.0000 | 11.0000 |
| cat_popular | g_cat_clean | matched | 0.1 | 0.9250 | 0.9231 | 0.4750 | 11.0000 | 5.0000 |
| cat_popular | d_attr_count_g1_s05_clean | mismatched | 0.05 | 0.9100 | 0.9082 | 0.4800 | 10.0000 | 7.0000 |
| cat_popular | g_rel_clean | mismatched | 0.05 | 0.9100 | 0.9062 | 0.4600 | 10.0000 | 7.0000 |
| cat_popular | d_cat_hard_g1_s05_clean | matched | 0.05 | 0.9100 | 0.9053 | 0.4500 | 9.0000 | 6.0000 |
| cat_popular | d_rel_contact_g1_s05_clean | mismatched | 0.05 | 0.9050 | 0.9045 | 0.4950 | 11.0000 | 9.0000 |
| cat_popular | d_attr_color_g1_s05_clean | mismatched | 0.05 | 0.9050 | 0.9016 | 0.4650 | 12.0000 | 10.0000 |
| cat_popular | d_rel_spatial_g1_s05_clean | mismatched | 0.05 | 0.8950 | 0.8912 | 0.4650 | 8.0000 | 8.0000 |
| cat_popular | g_all_clean | global | 0.1 | 0.8900 | 0.8866 | 0.4700 | 8.0000 | 9.0000 |
| cat_popular | g_attr_clean | mismatched | 0.25 | 0.8800 | 0.8788 | 0.4900 | 9.0000 | 12.0000 |
| cat_random | g_rel_clean | mismatched | 0.05 | 0.9500 | 0.9485 | 0.4700 | 11.0000 | 2.0000 |
| cat_random | g_all_clean | global | 0.1 | 0.9400 | 0.9388 | 0.4800 | 12.0000 | 5.0000 |
| cat_random | d_attr_color_g1_s05_clean | mismatched | 0.25 | 0.9350 | 0.9347 | 0.4950 | 15.0000 | 9.0000 |
| cat_random | d_rel_contact_g1_s05_clean | mismatched | 0.5 | 0.9350 | 0.9333 | 0.4750 | 15.0000 | 9.0000 |
| cat_random | d_rel_spatial_g1_s05_clean | mismatched | 0.5 | 0.9350 | 0.9333 | 0.4750 | 14.0000 | 8.0000 |
| cat_random | d_attr_count_g1_s05_clean | mismatched | 0.05 | 0.9350 | 0.9326 | 0.4650 | 13.0000 | 7.0000 |
| cat_random | g_attr_clean | mismatched | 0.25 | 0.9350 | 0.9326 | 0.4650 | 12.0000 | 6.0000 |
| cat_random | d_cat_hard_g1_s05_clean | matched | 0.5 | 0.9300 | 0.9278 | 0.4700 | 13.0000 | 8.0000 |
| cat_random | g_cat_clean | matched | 0.5 | 0.9200 | 0.9184 | 0.4800 | 12.0000 | 9.0000 |
| rel_contact | d_attr_count_g1_s05_clean | mismatched | 0.1 | 0.6950 | 0.7550 | 0.7450 | 33.0000 | 20.0000 |
| rel_contact | d_rel_spatial_g1_s05_clean | mismatched | 0.25 | 0.6900 | 0.7438 | 0.7100 | 31.0000 | 19.0000 |
| rel_contact | d_cat_hard_g1_s05_clean | mismatched | 0.1 | 0.6800 | 0.7355 | 0.7100 | 34.0000 | 24.0000 |
| rel_contact | g_all_clean | global | 0.5 | 0.6850 | 0.7342 | 0.6850 | 33.0000 | 22.0000 |
| rel_contact | g_rel_clean | mismatched | 0.1 | 0.6800 | 0.7333 | 0.7000 | 36.0000 | 26.0000 |
| rel_contact | g_cat_clean | mismatched | 0.25 | 0.6650 | 0.7243 | 0.7150 | 35.0000 | 28.0000 |
| rel_contact | d_rel_contact_g1_s05_clean | matched | 0.25 | 0.6750 | 0.7210 | 0.6650 | 35.0000 | 26.0000 |
| rel_contact | g_attr_clean | mismatched | 0.5 | 0.6500 | 0.7131 | 0.7200 | 29.0000 | 25.0000 |
| rel_contact | d_attr_color_g1_s05_clean | mismatched | 0.5 | 0.6400 | 0.7097 | 0.7400 | 34.0000 | 32.0000 |
| rel_spatial | d_attr_color_g1_s05_clean | mismatched | 0.1 | 0.6200 | 0.6960 | 0.7500 | 37.0000 | 26.0000 |
| rel_spatial | d_attr_count_g1_s05_clean | mismatched | 0.5 | 0.6050 | 0.6902 | 0.7750 | 37.0000 | 29.0000 |
| rel_spatial | g_all_clean | global | 0.25 | 0.6150 | 0.6883 | 0.7350 | 35.0000 | 25.0000 |
| rel_spatial | g_attr_clean | mismatched | 0.25 | 0.6050 | 0.6776 | 0.7250 | 32.0000 | 24.0000 |
| rel_spatial | g_cat_clean | mismatched | 0.1 | 0.5950 | 0.6773 | 0.7550 | 35.0000 | 29.0000 |
| rel_spatial | d_rel_contact_g1_s05_clean | mismatched | 0.5 | 0.5700 | 0.6767 | 0.8300 | 24.0000 | 23.0000 |
| rel_spatial | d_cat_hard_g1_s05_clean | mismatched | 0.25 | 0.5800 | 0.6744 | 0.7900 | 33.0000 | 30.0000 |
| rel_spatial | d_rel_spatial_g1_s05_clean | matched | 0.5 | 0.5750 | 0.6743 | 0.8050 | 27.0000 | 25.0000 |
| rel_spatial | g_rel_clean | mismatched | 0.25 | 0.5850 | 0.6693 | 0.7550 | 34.0000 | 30.0000 |

## 6. Controls
- Shuffle subtype control not available yet.
- Raw vs clean ablation not available yet.

## 7. Key Conclusion
- Best matched F1 (0.9278) does not beat best mismatched F1 (0.9485); subtype-specific experts are not established yet.

## 8. Concrete Next Actions
- If attr_color is the first clear matched winner, expand color data and run a larger GQA/MME-color sanity pass.
- If attr_count is positive, run an MME-count limit sweep next.
- If rel_spatial is positive, expand spatial relation data before touching rel_contact.
- If rel_contact stays weak, inspect relation label quality and counterfact mappings before adding more samples.
- If all subtype vectors are weak, revisit activation definition and hook timing rather than only tuning alpha/SVD.
