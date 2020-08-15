# BLEU implementation in joeynmet does not support n-gram BLEU, which is needed for comparison with reference paper

def bleu(output, truth_list, n):
    """
    compute belu score for ONE single output sentence
    :param output: String of output sentence
    :param truth_list: List of strings of reference (true) sentences
    :param n: BLEU-n to compute
    :return: 0<=score<=1
    """

    correct_sum = 0
    correct_length = 0  # Length of output, sanitized by all paddings '<pad>' and sequence tags '<s>', '</s>'
    split = output.split(" ")
    for i in range(len(split) - (n - 1)):
        token = ' '.join(split[i: i + n])
        if "<pad>" not in token and "<s>" not in token and "</s>" not in token:
            if ' ' + token + ' ' in ' # '.join(truth_list):
                correct_sum += 1
            correct_length += 1

    return correct_sum / correct_length


def test_bleu():
    bl = bleu("<s> hel lo i am super <pad> </s>", ["hello abc gr supger", "hello im gr super "], 1)
    print(bl)


if __name__ == '__main__':
    pass
