import json

def relevance_score(expected_title, expected_author,
                    candidate_title, candidate_author,
                    num_ratings):

    exp_title = expected_title.strip()
    exp_author = expected_author.strip()

    cand_title = candidate_title.strip()
    cand_author = candidate_author.strip()

    exp_tokens = set(exp_title.split())
    cand_tokens = set(cand_title.split())

    overlap = len(exp_tokens & cand_tokens)
    overlap_ratio = overlap / len(exp_tokens) if exp_tokens else 0

    author_match = (
        exp_author != "" and
        exp_author == cand_author
    )

    exact_title = exp_title == cand_title

    # 3 = exact title + exact author
    if exact_title and author_match:
        return 3

    # 2 = exact title OR strong overlap
    elif (
        (author_match and overlap_ratio >= 0.5)
        or
        (overlap_ratio >= 0.7 and num_ratings >= 1000)
    ):
        return 2

    # 1 = weak overlap
    elif (
        overlap > 0
        or author_match
    ):
        return 1

    # 0 = irrelevant
    else:
        return 0


# load source file
with open("v4experiment_results_alpha060.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# process each query
for item in data:

    expected_title = item["title"]
    expected_author = item["author"]

    for source_type, candidates in item["candidates"].items():

        for cand in candidates:

            score = relevance_score(
                expected_title,
                expected_author,
                cand["title"],
                cand["author"],
                cand.get("numRatings", 0)
            )

            cand["relevance_score"] = score


# save output
with open("ground_truth_scored.json", "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print("Done. Saved to ground_truth_scored.json")