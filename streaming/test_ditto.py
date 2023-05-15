import matcher

matcher.run_matcher(task="Structured/Beer", input_path="data/er_magellan/Structured/Beer/test.txt", output_path="output/output_small_test_fair.jsonl", lm="roberta", checkpoint_path="checkpoints/")