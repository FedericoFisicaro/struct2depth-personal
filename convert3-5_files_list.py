path_file_list_3 = "../../3/DNet/splits/umonsH3/test_files.txt"
split = "H3"
save_dir = "test_files_list/"


f = open(path_file_list_3,'r')
f_out = open(save_dir + split+ "_test_files.txt","w")

for line in f.readlines() :
    f_out.write(line.split()[0] + "\n")

f.close()
f_out.close()