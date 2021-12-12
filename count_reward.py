score_counter = 0
iter = 1

with open('clean_output_newnn.txt') as clean_file:
    with open('reward_output_newnn.txt', 'w') as reward_file:
        line = clean_file.readline()
        while line != "":
            reward = float(line.split(",")[3].split(": ")[1])
            #print(reward)
            if reward == 1:
                score_counter += 1
            if reward == -1:
                score_counter = 0
            reward_file.write(str(iter)+","+str(float(line.split(",")[4].split(": ")[1].strip()))+","+str(score_counter)+"\n")
            iter += 1
            line = clean_file.readline()
