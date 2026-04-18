"""Static category metadata used to build the COCO-scale negative bank."""

from __future__ import annotations

from typing import Final


def _flatten_group_categories(group_specs: dict[str, dict[str, object]]) -> list[str]:
    """Flatten category names from grouped specs while preserving declaration order."""

    ordered: list[str] = []
    for spec in group_specs.values():
        for category in spec["categories"]:
            ordered.append(str(category))
    return ordered


COCO80_GROUP_SPECS: Final[dict[str, dict[str, object]]] = {
    "human": {"supercategory": "person", "categories": ["person"]},
    "vehicle_cycle": {"supercategory": "vehicle", "categories": ["bicycle", "motorcycle"]},
    "vehicle_road": {"supercategory": "vehicle", "categories": ["car", "bus", "train", "truck"]},
    "vehicle_air_water": {"supercategory": "vehicle", "categories": ["airplane", "boat"]},
    "outdoor_fixture": {
        "supercategory": "outdoor",
        "categories": ["traffic light", "fire hydrant", "stop sign", "parking meter", "bench"],
    },
    "animal_pet": {"supercategory": "animal", "categories": ["bird", "cat", "dog"]},
    "animal_farm": {"supercategory": "animal", "categories": ["horse", "sheep", "cow"]},
    "animal_wild": {"supercategory": "animal", "categories": ["elephant", "bear", "zebra", "giraffe"]},
    "accessory_carry": {"supercategory": "accessory", "categories": ["backpack", "handbag", "suitcase"]},
    "accessory_wear": {"supercategory": "accessory", "categories": ["umbrella", "tie"]},
    "sports_flying": {"supercategory": "sports", "categories": ["frisbee", "sports ball", "kite"]},
    "sports_board": {"supercategory": "sports", "categories": ["skis", "snowboard", "skateboard", "surfboard"]},
    "sports_strike": {
        "supercategory": "sports",
        "categories": ["baseball bat", "baseball glove", "tennis racket"],
    },
    "kitchen_drinkware": {"supercategory": "kitchen", "categories": ["bottle", "wine glass", "cup"]},
    "kitchen_tableware": {"supercategory": "kitchen", "categories": ["fork", "knife", "spoon", "bowl"]},
    "food_fruit": {"supercategory": "food", "categories": ["banana", "apple", "orange"]},
    "food_vegetable": {"supercategory": "food", "categories": ["broccoli", "carrot"]},
    "food_prepared": {
        "supercategory": "food",
        "categories": ["sandwich", "hot dog", "pizza", "donut", "cake"],
    },
    "furniture_seating": {"supercategory": "furniture", "categories": ["chair", "couch"]},
    "furniture_room": {
        "supercategory": "furniture",
        "categories": ["potted plant", "bed", "dining table", "toilet"],
    },
    "electronic_screen": {"supercategory": "electronic", "categories": ["tv", "laptop", "cell phone"]},
    "electronic_accessory": {"supercategory": "electronic", "categories": ["mouse", "remote", "keyboard"]},
    "appliance_kitchen": {
        "supercategory": "appliance",
        "categories": ["microwave", "oven", "toaster", "sink", "refrigerator"],
    },
    "indoor_misc": {
        "supercategory": "indoor",
        "categories": ["book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"],
    },
}

EXTRA_COMPAT_GROUP_SPECS: Final[dict[str, dict[str, object]]] = {
    "mock_animal": {"supercategory": "animal", "categories": ["rabbit", "fox"]},
    "mock_indoor": {"supercategory": "indoor", "categories": ["mat"]},
    "mock_sports": {"supercategory": "sports", "categories": ["ball"]},
}

SEMANTIC_GROUP_FALLBACKS: Final[dict[str, list[str]]] = {
    "human": ["accessory_carry", "accessory_wear", "indoor_misc"],
    "vehicle_cycle": ["vehicle_road", "vehicle_air_water", "sports_board"],
    "vehicle_road": ["vehicle_cycle", "vehicle_air_water", "outdoor_fixture"],
    "vehicle_air_water": ["vehicle_road", "vehicle_cycle"],
    "outdoor_fixture": ["vehicle_road", "furniture_seating", "human"],
    "animal_pet": ["animal_farm", "animal_wild", "mock_animal"],
    "animal_farm": ["animal_pet", "animal_wild", "mock_animal"],
    "animal_wild": ["animal_pet", "animal_farm", "mock_animal"],
    "accessory_carry": ["accessory_wear", "human", "indoor_misc"],
    "accessory_wear": ["accessory_carry", "human", "indoor_misc"],
    "sports_flying": ["sports_board", "sports_strike", "mock_sports"],
    "sports_board": ["sports_flying", "sports_strike", "vehicle_cycle"],
    "sports_strike": ["sports_flying", "sports_board", "mock_sports"],
    "kitchen_drinkware": ["kitchen_tableware", "food_prepared", "appliance_kitchen"],
    "kitchen_tableware": ["kitchen_drinkware", "food_prepared", "appliance_kitchen"],
    "food_fruit": ["food_vegetable", "food_prepared", "kitchen_tableware"],
    "food_vegetable": ["food_fruit", "food_prepared", "kitchen_tableware"],
    "food_prepared": ["food_fruit", "food_vegetable", "kitchen_tableware"],
    "furniture_seating": ["furniture_room", "indoor_misc", "outdoor_fixture", "mock_indoor"],
    "furniture_room": ["furniture_seating", "indoor_misc", "appliance_kitchen", "mock_indoor"],
    "electronic_screen": ["electronic_accessory", "appliance_kitchen", "indoor_misc"],
    "electronic_accessory": ["electronic_screen", "accessory_carry", "indoor_misc"],
    "appliance_kitchen": ["kitchen_drinkware", "kitchen_tableware", "furniture_room", "electronic_screen"],
    "indoor_misc": ["furniture_room", "furniture_seating", "accessory_carry", "electronic_screen"],
    "mock_animal": ["animal_pet", "animal_farm", "animal_wild"],
    "mock_indoor": ["furniture_room", "furniture_seating", "indoor_misc"],
    "mock_sports": ["sports_flying", "sports_board", "sports_strike"],
}

CATEGORY_SEED_NEGATIVES: Final[dict[str, list[str]]] = {
    "person": ["teddy bear", "backpack", "umbrella", "handbag", "suitcase", "bench"],
    "bicycle": ["motorcycle", "skateboard", "car", "bus", "train", "truck"],
    "motorcycle": ["bicycle", "car", "truck", "bus", "train", "skateboard"],
    "car": ["truck", "bus", "train", "motorcycle", "bicycle", "boat"],
    "truck": ["car", "bus", "train", "motorcycle", "boat", "refrigerator"],
    "boat": ["airplane", "car", "bus", "truck", "surfboard", "skateboard"],
    "cat": ["dog", "fox", "rabbit", "bird", "bear", "horse"],
    "dog": ["cat", "fox", "rabbit", "bird", "bear", "horse"],
    "sports ball": ["frisbee", "kite", "baseball glove", "apple", "orange", "donut"],
    "ball": ["frisbee", "kite", "sports ball", "apple", "orange", "donut"],
    "chair": ["couch", "bench", "bed", "toilet", "dining table", "potted plant"],
    "tv": ["laptop", "cell phone", "microwave", "oven", "refrigerator", "book"],
    "book": ["clock", "vase", "scissors", "remote", "keyboard", "toothbrush"],
    "mat": ["chair", "couch", "bed", "dining table", "toilet", "potted plant"],
}

COCO80_CATEGORY_NAMES: Final[list[str]] = _flatten_group_categories(COCO80_GROUP_SPECS)
SUPPORTED_NEGATIVE_CATEGORY_NAMES: Final[list[str]] = COCO80_CATEGORY_NAMES + _flatten_group_categories(
    EXTRA_COMPAT_GROUP_SPECS
)
