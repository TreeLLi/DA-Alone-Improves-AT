'''
print supporting multiprocessing distribution and save in the file

'''

rank = None
def set_rank(r):
    global rank
    rank = r

WIDTH = 90
SPLIT = '-'*WIDTH


def print_head(head):
    margin = (WIDTH - len(head)) // 2
    head = ' '*margin + head + ' '*margin
    print("{}\n{}\n{}".format(SPLIT, head.upper(), SPLIT))

def word(k, v):
    k += ':'
    msg = "{:15}"
    if isinstance(v, bool):
        v = 'True' if v else 'False'
    elif isinstance(v, float):
        if v < 1e-4:
            v = "{:.3e}".format(v)
        else:
            nfloats = len(str(v).split('.')[-1])
            v = "{:.4f}".format(v) if nfloats > 4 else str(v)
    else:
        v = str(v)
    msg += '{:13}'
    return msg.format(k, v)
            
def print_lines(words):
    line = ""
    for i, (k, v) in enumerate(words.items()):
        if i % 3 == 0 and line != "":
            print(line)
            line = ""
        line += word(k, v)
        if (i+1) % 3 != 0:
            line += ' | '
    if line != "":
        print(line)
    
def dprint(head=None, **kwargs):
    if rank is not None and rank != 0:
        return

    if head is not None:
        print_head(head)

    print_lines(kwargs)

def sprint(msg, split=False):
    if rank is None or rank==0:
        if split:
            msg = "{}\n{}\n{}".format(SPLIT, msg, SPLIT)
        print(msg)
