def get_script(s, script='sup'):
    if script == 'sup':
        return s.translate(s.maketrans('0123456789', '⁰¹²³⁴⁵⁶⁷⁸⁹'))
    if script == 'sub':
        return s.translate(s.maketrans('0123456789', '₀₁₂₃₄₅₆₇₈₉'))
    return s


def polynom2str(a, x='x', script='sup'):
    s = ''
    frm = 0
    while a[frm] == 0: frm += 1
    first_done = False
    for i in range(frm, len(a)):
        if a[i] == 0: continue
        deg = len(a) - i - 1
        if first_done:
            s += ('+ ' if a[i] > 0 else '- ')
        elif a[i] < 0:
            s += '-'
        first_done = True
        if abs(a[i]) != 1 or deg == 0:
            s += '%g' % abs(a[i])
        if script == 'sup':
            if deg > 0:
                s += x
                if deg != 1:
                    s += get_script(str(deg), script)
                s += ' '
        else:
            s += x + get_script(str(i + 1), script) + ' '
    return s if s != '' else '0'


def print_system(a, b):
    strings = []
    for i in range(len(a)):
        strings.append('{')
        strings.append(polynom2str(a[i], script='sub'))
        strings.append('= ')
        strings.append(b[i])
        strings.append('}\n')
    return strings

def print_x_solution(x, sym='x'):
    for i in range(1, len(x) + 1):
        print(f'{sym}{get_script(str(i), "sub")} = %g;' % round(x[i - 1], 3), end='  ')
    print()


def print_vec_solution(x, sym='x'):
    for i in range(1, len(x) + 1):
        print(f'{sym}{get_script(str(i), "sub")} = {x[i - 1]};')
    print()


def print_comp_solution(x, sym='x'):
    for i in range(1, len(x) + 1):
        s = str(round(x[i - 1][0], 4))
        if x[i - 1][1] != 0:
            if x[i - 1][1] < 0:
                s += ' - '
            else:
                s += ' + '
            s += str(round(abs(x[i - 1][1]), 4)) + ' i'
        print(f'{sym}{get_script(str(i), "sub")} = {s};')
    print()