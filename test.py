import sample as smp
import generate_title as gnrt


def clear_tail(data, split_char=' ', tail_length=1):
    return split_char.join(data.split(split_char)[:-tail_length])


sample_txt = smp.main(gnrt.get_title(), 64, print_res=False)
final_txt = clear_tail(sample_txt, ':')
print(final_txt)
