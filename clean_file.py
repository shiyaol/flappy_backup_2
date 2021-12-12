file1 = open('out_slurmtest 2','r')
#file2 = open('clean_output.txt', 'w')
lines = file1.readlines()
with open('clean_output_newnn.txt', 'w') as out:
    for line in lines:
        if line.find('Q-value') != -1:
            out.write(line)

file1.close()