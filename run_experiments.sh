for primes in 0 1 2 4 8
do
	for seed in 10 20 30
	do
		for dataset in openbookqa commonsense_qa mmlu
		do
			for model in curie davinci davinci-instruct-beta facebook/opt-30b google/flan-t5-xxl
			do
				for q_type in no_answer_choices string_answer_choices enumerated_answer_choices
				do
					python main.py \
						--num-primes=$primes \
						--model=$model \
						--random-seed=$seed \
						--dataset=$dataset \
						--prompt-format=$q_type
				done
			done
		done
	done
done
