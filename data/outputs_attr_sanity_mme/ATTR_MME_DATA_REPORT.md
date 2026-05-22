# Attribute MME Data Report

- Benchmark root: `/home/huiwei/sy/halludata/data/benchmarks/mme_hallucination`
- Image root: `/home/huiwei/sy/halludata/data/benchmarks/mme_hallucination/images`

## Overview

| category | n | yes | no | invalid | unique_images | missing_images | duplicate_pairs | parser_risks |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| existence | 60 | 30 | 30 | 0 | 60 | 0 | 0 | 120 |
| count | 60 | 30 | 30 | 0 | 60 | 0 | 0 | 120 |
| color | 60 | 30 | 30 | 0 | 60 | 0 | 0 | 120 |
| position | 60 | 30 | 30 | 0 | 60 | 0 | 0 | 120 |

## existence

- Samples: `60`
- Label counts: yes=`30`, no=`30`, invalid=`0`
- Missing images: `0`
- Duplicate image/question pairs: `0`
- Parser risk counts: `{"not_question_mark_terminated": 60, "question_contains_yes_or_no": 60}`

Examples:

```json
[
  {
    "sample_id": "mme_existence_00000",
    "image": "existence/mme_existence_00000_000000494427.jpg",
    "image_exists": true,
    "question": "Is there a laptop in this image? Please answer yes or no.",
    "label": "yes"
  },
  {
    "sample_id": "mme_existence_00001",
    "image": "existence/mme_existence_00001_000000494427.jpg",
    "image_exists": true,
    "question": "Is there a potted plant in this image? Please answer yes or no.",
    "label": "no"
  },
  {
    "sample_id": "mme_existence_00002",
    "image": "existence/mme_existence_00002_000000015517.jpg",
    "image_exists": true,
    "question": "Is there a bus in this image? Please answer yes or no.",
    "label": "yes"
  },
  {
    "sample_id": "mme_existence_00003",
    "image": "existence/mme_existence_00003_000000015517.jpg",
    "image_exists": true,
    "question": "Is there a cow in this image? Please answer yes or no.",
    "label": "no"
  },
  {
    "sample_id": "mme_existence_00004",
    "image": "existence/mme_existence_00004_000000009590.jpg",
    "image_exists": true,
    "question": "Is there a bottle in this image? Please answer yes or no.",
    "label": "yes"
  }
]
```

Parser risk examples:

```json
[
  {
    "risk": "question_contains_yes_or_no",
    "question": "Is there a laptop in this image? Please answer yes or no.",
    "label": "yes"
  },
  {
    "risk": "not_question_mark_terminated",
    "question": "Is there a laptop in this image? Please answer yes or no.",
    "label": "yes"
  },
  {
    "risk": "question_contains_yes_or_no",
    "question": "Is there a potted plant in this image? Please answer yes or no.",
    "label": "no"
  },
  {
    "risk": "not_question_mark_terminated",
    "question": "Is there a potted plant in this image? Please answer yes or no.",
    "label": "no"
  },
  {
    "risk": "question_contains_yes_or_no",
    "question": "Is there a bus in this image? Please answer yes or no.",
    "label": "yes"
  }
]
```

## count

- Samples: `60`
- Label counts: yes=`30`, no=`30`, invalid=`0`
- Missing images: `0`
- Duplicate image/question pairs: `0`
- Parser risk counts: `{"not_question_mark_terminated": 60, "question_contains_yes_or_no": 60}`

Examples:

```json
[
  {
    "sample_id": "mme_count_00000",
    "image": "count/mme_count_00000_000000470121.jpg",
    "image_exists": true,
    "question": "Is there only one bottle in the image? Please answer yes or no.",
    "label": "yes"
  },
  {
    "sample_id": "mme_count_00001",
    "image": "count/mme_count_00001_000000470121.jpg",
    "image_exists": true,
    "question": "Is there two bottles in the image? Please answer yes or no.",
    "label": "no"
  },
  {
    "sample_id": "mme_count_00002",
    "image": "count/mme_count_00002_000000430286.jpg",
    "image_exists": true,
    "question": "Are there three remotes in this image? Please answer yes or no.",
    "label": "yes"
  },
  {
    "sample_id": "mme_count_00003",
    "image": "count/mme_count_00003_000000430286.jpg",
    "image_exists": true,
    "question": "Are there only two remotes in this image? Please answer yes or no.",
    "label": "no"
  },
  {
    "sample_id": "mme_count_00004",
    "image": "count/mme_count_00004_000000301867.jpg",
    "image_exists": true,
    "question": "Are there three people appear in this image? Please answer yes or no.",
    "label": "yes"
  }
]
```

Parser risk examples:

```json
[
  {
    "risk": "question_contains_yes_or_no",
    "question": "Is there only one bottle in the image? Please answer yes or no.",
    "label": "yes"
  },
  {
    "risk": "not_question_mark_terminated",
    "question": "Is there only one bottle in the image? Please answer yes or no.",
    "label": "yes"
  },
  {
    "risk": "question_contains_yes_or_no",
    "question": "Is there two bottles in the image? Please answer yes or no.",
    "label": "no"
  },
  {
    "risk": "not_question_mark_terminated",
    "question": "Is there two bottles in the image? Please answer yes or no.",
    "label": "no"
  },
  {
    "risk": "question_contains_yes_or_no",
    "question": "Are there three remotes in this image? Please answer yes or no.",
    "label": "yes"
  }
]
```

## color

- Samples: `60`
- Label counts: yes=`30`, no=`30`, invalid=`0`
- Missing images: `0`
- Duplicate image/question pairs: `0`
- Parser risk counts: `{"not_question_mark_terminated": 60, "question_contains_yes_or_no": 60}`

Examples:

```json
[
  {
    "sample_id": "mme_color_00000",
    "image": "color/mme_color_00000_000000338560.jpg",
    "image_exists": true,
    "question": "Is there a blue and yellow fire hydrant in the image? Please answer yes or no.",
    "label": "yes"
  },
  {
    "sample_id": "mme_color_00001",
    "image": "color/mme_color_00001_000000338560.jpg",
    "image_exists": true,
    "question": "Is there a blue and orange fire hydrant in the image? Please answer yes or no.",
    "label": "no"
  },
  {
    "sample_id": "mme_color_00002",
    "image": "color/mme_color_00002_000000047112.jpg",
    "image_exists": true,
    "question": "Is there a white plate in the image? Please answer yes or no.",
    "label": "yes"
  },
  {
    "sample_id": "mme_color_00003",
    "image": "color/mme_color_00003_000000047112.jpg",
    "image_exists": true,
    "question": "Is there a yellow plate in the image? Please answer yes or no.",
    "label": "no"
  },
  {
    "sample_id": "mme_color_00004",
    "image": "color/mme_color_00004_000000512929.jpg",
    "image_exists": true,
    "question": "Are there any green beans in the image? Please answer yes or no.",
    "label": "yes"
  }
]
```

Parser risk examples:

```json
[
  {
    "risk": "question_contains_yes_or_no",
    "question": "Is there a blue and yellow fire hydrant in the image? Please answer yes or no.",
    "label": "yes"
  },
  {
    "risk": "not_question_mark_terminated",
    "question": "Is there a blue and yellow fire hydrant in the image? Please answer yes or no.",
    "label": "yes"
  },
  {
    "risk": "question_contains_yes_or_no",
    "question": "Is there a blue and orange fire hydrant in the image? Please answer yes or no.",
    "label": "no"
  },
  {
    "risk": "not_question_mark_terminated",
    "question": "Is there a blue and orange fire hydrant in the image? Please answer yes or no.",
    "label": "no"
  },
  {
    "risk": "question_contains_yes_or_no",
    "question": "Is there a white plate in the image? Please answer yes or no.",
    "label": "yes"
  }
]
```

## position

- Samples: `60`
- Label counts: yes=`30`, no=`30`, invalid=`0`
- Missing images: `0`
- Duplicate image/question pairs: `0`
- Parser risk counts: `{"not_question_mark_terminated": 60, "question_contains_yes_or_no": 60}`

Examples:

```json
[
  {
    "sample_id": "mme_position_00000",
    "image": "position/mme_position_00000_000000472046.jpg",
    "image_exists": true,
    "question": "Is the pineapple on the left of the pot in the image? Please answer yes or no.",
    "label": "yes"
  },
  {
    "sample_id": "mme_position_00001",
    "image": "position/mme_position_00001_000000472046.jpg",
    "image_exists": true,
    "question": "Is the pineapple on the right of the pot in the image? Please answer yes or no.",
    "label": "no"
  },
  {
    "sample_id": "mme_position_00002",
    "image": "position/mme_position_00002_000000067213.jpg",
    "image_exists": true,
    "question": "Is the dog above the pool in the image? Please answer yes or no.",
    "label": "yes"
  },
  {
    "sample_id": "mme_position_00003",
    "image": "position/mme_position_00003_000000067213.jpg",
    "image_exists": true,
    "question": "Is the dog under the pool in the image? Please answer yes or no.",
    "label": "no"
  },
  {
    "sample_id": "mme_position_00004",
    "image": "position/mme_position_00004_000000530162.jpg",
    "image_exists": true,
    "question": "Is the big red and black umbrella on the top of people? Please answer yes or no.",
    "label": "yes"
  }
]
```

Parser risk examples:

```json
[
  {
    "risk": "question_contains_yes_or_no",
    "question": "Is the pineapple on the left of the pot in the image? Please answer yes or no.",
    "label": "yes"
  },
  {
    "risk": "not_question_mark_terminated",
    "question": "Is the pineapple on the left of the pot in the image? Please answer yes or no.",
    "label": "yes"
  },
  {
    "risk": "question_contains_yes_or_no",
    "question": "Is the pineapple on the right of the pot in the image? Please answer yes or no.",
    "label": "no"
  },
  {
    "risk": "not_question_mark_terminated",
    "question": "Is the pineapple on the right of the pot in the image? Please answer yes or no.",
    "label": "no"
  },
  {
    "risk": "question_contains_yes_or_no",
    "question": "Is the dog above the pool in the image? Please answer yes or no.",
    "label": "yes"
  }
]
```
