# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 11:10:52 2020

@author: 段宇飞
"""
from math import gcd, sqrt, ceil
from Crypto.Util.number import *
import base64
import requests
import gmpy2
import re
import json
import random, time

morse = {}
morse['01'] = 'a'
morse['1000'] = 'b'
morse['1010'] = 'c'
morse['100'] = 'd'
morse['0'] = 'e'
morse['0010'] = 'f'
morse['110'] = 'g'
morse['0000'] = 'h'
morse['00'] = 'i'
morse['0111'] = 'j'
morse['101'] = 'k'
morse['0100'] = 'l'
morse['11'] = 'm'
morse['10'] = 'n'
morse['111'] = 'o'
morse['0110'] = 'p'
morse['1101'] = 'q'
morse['010'] = 'r'
morse['000'] = 's'
morse['1'] = 't'
morse['001'] = 'u'
morse['0001'] = 'v'
morse['011'] = 'w'
morse['1001'] = 'x'
morse['1011'] = 'y'
morse['1100'] = 'z'
morse['01111'] = '1'
morse['00111'] = '2'
morse['00011'] = '3'
morse['00001'] = '4'
morse['00000'] = '5'
morse['10000'] = '6'
morse['11000'] = '7'
morse['11100'] = '8'
morse['11110'] = '9'
morse['11111'] = '0'


class Point:
    eps = 1e-8
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __str__(self):
        return str(tuple((self.x, self.y)))
    
    def __repr__(self):
        return self.__str__()
    
    def __add__(self, other):
        return Point(self.x + other.x, self.y + other.y)
    
    def __sub__(self, other):
        return Point(self.x - other.x, self.y - other.y)

class ECCPoint(Point):
    
    def __init__(self, a, b, p, O = False, x = None, y = None):
        self.a = a
        self.b = b
        self.p = p
        self.O = O
        if x != None:
            self.x = x
        if y != None:
            self.y = y
    
    def addPoint(self, x, y):
        tmp = ECCPoint(self.a, self.b, self.p, x = x, y = y)
        return tmp
    
    def __str__(self):
        if self.O:
            return "O"
        else:
            return str(tuple((self.x, self.y)))
    
    def __repr__(self):
        return self.__str__()
    
    def __add__(self, other):
        if other.O:
            return self
        x1, y1 = self.x, self.y
        x2, y2 = other.x, other.y
        if x1 == x2 and (y1 + y2) % self.p == 0:
            return ECCPoint(self.a, self.b, self.p, O = True, x = 0, y = 0)
        if x1 == x2 and y1 == y2:
            lamta = (3 * x1 * x1 + self.a) * inverse((2 * y1), self.p) % self.p
        else:
            lamta = (y2 - y1) * inverse(x2 - x1, self.p) % self.p
        x3 = lamta * lamta - x1 - x2
        y3 = lamta * (x1 - x3) - y1
        
        return ECCPoint(self.a, self.b, self.p, x = x3 % self.p, y = y3 % self.p)
    
    
    def __mul__(self, other):
        if other == 1:
            return self
        tmp = ECCPoint(self.a, self.b, self.p,x = self.x, y = self.y)
        for i in range(other - 1):
            tmp = tmp + self
        
        return tmp

def phi(m):
    if isPrime(m):
        return m - 1
    ret = 0
    for i in range(m):
        if gcd(i + 1, m) == 1:
            ret += 1
    return ret

def order(a, m):
    if gcd(a, m) != 1:
        return -1
    else:
        tmp = phi(m)
        for i in range(tmp):
            if a ** (i + 1) % m == 1:
                return i + 1

def powmod(x, y, m):
    ret = 1
    for i in range(y):
        ret = ret * x % m
    return ret

def g(m):
    ret = []
    g = 0
    ph = phi(m)
    for i in range(m):
        if gcd(i + 1, m) == 1 and order(i + 1, m) == ph:
            g = i + 1
            break
    for i in range(ph):
        if gcd(i + 1, ph) == 1:
            ret.append(powmod(g, i+1, m))
    ret.sort()
    return tuple(ret)

def getPrimes(N):
    ret = []
    for i in range(2, N + 1):
        if isPrime(i):
            ret.append(i)
    return ret

def caesar(stringstream, offset):
    ret = ""
    for each in stringstream:
        if not str.isalpha(each):
            ret += each
        elif str.islower(each):
            tmp = ord(each) + offset
            while tmp > 122:
                tmp -= 26
            while tmp < 97:
                tmp += 26
            ret += chr(tmp)
        else:
            tmp = ord(each) + offset
            while tmp > 90:
                tmp -= 26
            while tmp < 65:
                tmp += 26
            ret += chr(tmp)
    
    return ret
                
def Morse(cipertext, flag0 = '0', flag1 = '1', Split = ' '):
    cipertext = cipertext.replace(flag0, '0')
    cipertext = cipertext.replace(flag1, '1')
    cipertext = cipertext.split(Split)
    ret = ""
    for each in cipertext:
        try:
            ret += morse[each]
        except:
            ret += '*'
    return ret

def base64decode(s):
    return base64.b64decode(s).decode('utf-8')


def shiftleft(s, offset):
    return s[offset:] + s[:offset]

def roll(rolls, key, cipertext):
    tmp = []
    for each in key:
        tmp.append(rolls[each - 1])
    
    for i in range(len(cipertext)):
        p = tmp[i].find(cipertext[i])
        if p != 0:
            tmp[i] = shiftleft(tmp[i], p)
    
    for i in range(len(rolls[0])):
        for each in tmp:
            print(each[i], end = '')
        print("")

def num2str(num):
    tmp=str(hex(num))[2:]
    if len(tmp) % 2 == 0:
        pass
    else:
        tmp = '0' + tmp
    s = ''
    for i in range(0, len(tmp), 2):
        temp = tmp[i] + tmp[i + 1]
        s += chr(int(temp, 16))
    return s

def factordb(n, timeout = 10):
    url = "http://www.factordb.com/index.php?query=%d" % n
    try:
        res = requests.get(url, timeout = timeout)
    except:
        return None
    res.encoding = 'utf-8'
    ans = re.findall('color="(.*?)">(.*?)</font>', res.text, re.S)
    ret = []
    for i in range(len(ans)):
        ret.append(int(ans[i][1]))
    return ret
    
    

def RSADecode(ciphertext, p, q, e):
    if not isinstance(ciphertext, int):
        c = bytes_to_long(ciphertext)
    else:
        c = ciphertext
    n = p * q
    d = inverse(e, (p - 1) * (q - 1))
    m = pow(c, d, n)
    return long_to_bytes(m)

def CRT(Bs, Ms, Min = None):
    ans = 0
    l = len(Bs)
    m = 1
    for eachm in Ms:
        m *= eachm
    
    for i in range(l):
        ans = (ans + Bs[i] * (m // Ms[i]) * inverse(m // Ms[i], Ms[i])) % m
    
    if Min != None:
        while ans < Min:
            ans += m
    return ans

def eulerFactor(n):
    tmp = gmpy2.iroot(n, 2)[0] + 1
    while True:
        if gmpy2.iroot(tmp * tmp - n, 2)[1]:
            m = gmpy2.iroot(tmp * tmp - n, 2)[0]
            return (tmp - m, tmp + m)
        tmp += 1

def str_to_Cstr(s):
    ret = ""
    for each in s:
        ret += '\\x%s' % hex(ord(each))[2:]
    
    return ret

def rc4_decode(c, key):
    url = "http://tool.chacuo.net/cryptrc4"
    data = {}
    data['data'] = c
    data['type'] = 'rc4'
    data['arg'] = "p=%s_s=gb2312_t=1" % key
    
    res = requests.post(url, data = data)
    response = json.loads(res.text)
    
    return response['data'][0]

def Pollard_rho(n, seed = int(time.time())):
    
    random.seed(seed)
    def f(x):
        return (x * x + seed) % n
    
    x = 2
    y = f(2)
    
    while x != y:
        x = f(x)
        y = f(f(y))
        
        p = gcd(abs(y - x), n)
        if p > 1:
            return "Found Factor: %d" % p
    
    return "Failed"


def Pollard_P_1(n, B):              # for p-1 is smooth
    tmp = 2
    for i in range(2, B + 1):
        tmp = pow(tmp, i, n)
    
    g = gcd(tmp - 1, n)
    if g == 1:
        return "Failed"
    else:
        return "Found Factor: %d" % g

def Williams(n, B, A = 2):          # for p+1 is smooth
    pass

def RSAdecompose(n, phi):
    tmp = phi - 1
    s = 0
    while tmp % 2 == 0:
        s += 1
        tmp //= 2
    t = tmp
    
    A = 0
    I = 0
    
    find = False
    for a in range(2, n):
        for i in range(1, s + 1):
            if pow(a, pow(2, i - 1) * t, n) != 1 and pow(a, pow(2, i - 1) * t, n) != n - 1 and pow(a, pow(2,i) * t, n) == 1:
                A = a
                I = i
                find = True
                break
        if find:
            break
    
    if A == 0 and I == 0:
        return None
    
    p = gcd(pow(A, pow(2, I - 1) * t, n) - 1, n)
    q = n // p
    assert p * q == n
    
    return (p, q)

def WienerAttack(e, n):
    
    def rational_to_quotients(x, y):
        a = x // y
        quotients = [a]
        while a * y != x:
            x, y = y, x - a * y
            a = x // y
            quotients.append(a)
        return quotients

    def convergents_from_quotients(quotients):
        convergents = [(quotients[0], 1)]
        for i in range(2, len(quotients) + 1):
            quotients_partion = quotients[0:i]
            denom = quotients_partion[-1]  # 分母
            num = 1
            for _ in range(-2, -len(quotients_partion), -1):
                num, denom = denom, quotients_partion[_] * denom + num
            num += denom * quotients_partion[0]
            convergents.append((num, denom))
        return convergents
    
    quotients = rational_to_quotients(e, n)
    convergents = convergents_from_quotients(quotients)
    for (k, d) in convergents:
        if k and not (e * d - 1) % k:
            phi = (e * d - 1) // k
            # check if (x^2 - coef * x + n = 0) has integer roots
            coef = n - phi + 1
            delta = coef * coef - 4 * n
            if delta > 0 and gmpy2.iroot(delta, 2)[1] == True:
                print('d = ' + str(d))
                return d

import string

class str_generator:
    def __init__(self, charset, length, byte_mode = False):
        self.length = length
        self.charset = charset
        self.byte_mode = byte_mode
        if byte_mode:
            assert isinstance(charset, bytes)
    
    def __iter__(self):
        if self.byte_mode:
            self.string = long_to_bytes(self.charset[0]) * self.length
        else:   
            self.string = self.charset[0] * self.length
        return self
    
    def __next__(self):
        ret = self.string
        now = list(self.string)
        pointer = self.length - 1
        while now[pointer] == self.charset[-1] and pointer >= 0:
            now[pointer] = self.charset[0]
            pointer -= 1
            
        if pointer < 0:
            raise StopIteration
        else:
            now[pointer] = self.charset[self.charset.index(now[pointer]) + 1]
            if self.byte_mode:
                self.string = bytes(now)
            else:
                self.string = ''.join(now)
        return ret

def hashbreakn(hashcode, hashfunc, n, now, charset, format_string):
    if len(now) == n:
        if hashfunc((format_string % now).encode()).hexdigest() == hashcode:
            return now
        else:
            return False
    
    for each in charset:
        ret = hashbreakn(hashcode, hashfunc, n, now + each, charset, format_string)
        if ret:
            return ret

    return False

def hashbreak(hashcode, hashfunc, N = 0, charset = string.printable, format_string = "%s"):
    length = 1
    while length != N:
        print(length)
        ret = hashbreakn(hashcode, hashfunc, length, "", charset, format_string)
        if ret:
            return ret
        length += 1
    return None

def mergeCRT(b1, m1, b2, m2):
    g = gcd(m1, m2)
    m = m1 * m2 // g
    if gcd(m1 // g, m2 // g) != 1:
        return None
    b = (inverse(m1 // g, m2 // g) * (b2 - b1) // g) % (m2 // g) * m1 + b1
    return b, m

def exCRT(bs, ms, Min = None):
    l = len(bs)
    tmp = (bs[0], ms[0])
    for i in range(1, l):
        tmp = mergeCRT(tmp[0], tmp[1], bs[i], ms[i])
        if tmp == None:
            return None
    
    return tmp
    

def solve(a, b, c, realRoot = False):
    delta = b ** 2 - 4 * a * c
    if delta < 0:
        return None
    if realRoot:
        if delta == 0:
            return (-b / (2 * a), -b / (2 * a))
        tmp = sqrt(delta)
        return ((-b + tmp) / (2 * a), (-b - tmp) / (2 * a))
        
    tmp, check = gmpy2.iroot(delta, 2)
    if not check:
        return None
    
    return ((-b + tmp) // (2 * a), (-b - tmp) // (2 * a))

def discrete_log(g, y, p):
    m = int(ceil(sqrt(p - 1)))
    S = {pow(g, j ,p): j for j in range(m)}
    gs = pow(g, p - 1 - m, p)
    for i in range(m):
        if y in S:
            return i * m + S[y]
        y = y * gs % p
    return None

def AMM(x, r, m):
    
    '''
    get one solution for y ** e = x mod m
    '''
    
    x = x % m
    p = random.randint(1, m)
    while pow(p, (m - 1) // r, m) == 1:
        p = random.randint(1, m)
    
    t = 0
    s = m - 1
    while s % r == 0:
        t += 1
        s //= r
    k = 1
    while (k * s + 1) % r != 0:
        k += 1
    
    alpha = (k * s + 1) // r
    
    a = pow(p, pow(r, t - 1, m) * s % m, m)
    b = pow(x, r * alpha - 1, m)
    c = pow(p, s, m)
    h = 1
    
    for i in range(1, t):
        d = pow(b, pow(r, t - i - 1, m), m)
        if d == 1:
            j = 0
        else:
            j = -discrete_log(a, d, m)
        b = b * pow(pow(c, r, m), j, m) % m
        h = h * pow(c, j, m) % m
        c = pow(c, r, m)
    
    return pow(x, alpha, m) * h % m

def findAllPRoots(e, p):
    res = set()
    while len(res) < e:
        res.add(pow(random.randint(2, p - 1), (p - 1) // e, p))
    return res

def findAllSolutions(x, e, m):
    
    '''
    get all solutions for y ** e = x mod m
    '''
    
    proots = findAllPRoots(e, m)
    one_result = AMM(x, e, m)
    result = set()
    for root in proots:
        tmp = one_result * root % m
        assert pow(tmp, e, m) == x
        result.add(tmp)
    return list(result)


def xgcd(a,b):
    x, lastX = 0, 1
    y, lastY = 1, 0
    while (b != 0):
        q = a // b
        a, b = b, a % b
        x, lastX = lastX - q * x, x
        y, lastY = lastY - q * y, y
    return (lastX, lastY)

IC = 0.065
charset = string.ascii_letters
naturalLanguageP = [0.08167, 0.01492, 0.02782, 0.04253, 0.12702, 0.02228, 0.02015, 0.06094, 0.06966, 0.00153, 0.00772, 0.04025, 0.02406, 0.06749, 0.07507, 0.01929, 0.00095, 0.05987, 0.06327, 0.09056,0.02758, 0.00978, 0.02360, 0.00150, 0.01974, 0.00074]

def count_p(text):
    count = {}
    for each in text:
        if each not in count:
            count[each] = 1
        else:
            count[each] += 1
    return count

def get_ic(text):
    count = count_p(text)
    length = len(text)
    ic = 0
    for each in count:
        ic = ic + (count[each] - 1) * count[each]
    
    return ic / (length * (length - 1))

def split_text(text, length):
    strings = []
    len_text = len(text)
    for i in range(length):
        this = ""
        for j in range(i, len_text, length):
            this += text[j]
        strings.append(this)
        
    return strings

from math import inf

def get_key_length(text, max_length, min_length):
    
    min_diff = inf
    probably_length = 0
    
    for i in range(min_length, max_length + 1):
        
        strings = split_text(text, i)
        total_ic = 0
        for each in strings:
            total_ic += (get_ic(each) - IC) ** 2
        
        if total_ic < min_diff:
            (min_diff, probably_length) = (total_ic, i)
    
    return (min_diff, probably_length)


def break_key(text, length, decode_function, key_char):
    strings = split_text(text, length)
    key = ""
    for i in range(length):
        maxx = 0
        this_char = ''
        for each_char in key_char:
            plain_text = decode_function(strings[i], each_char).lower()
            pure_text = ""
            count = count_p(plain_text)
            this_p = []
            total = len(plain_text)
            for each in string.ascii_lowercase:
                if each not in count:
                    this_p.append(0)
                else:
                    this_p.append(count[each] / total)
            
            tmp = 0
            for j in range(26):
                tmp += this_p[j] * naturalLanguageP[j]
            if tmp > maxx:
                (maxx, this_char) = (tmp, each_char)
        
        key += this_char
    return key

def stradd(s1, s2, charset = string.ascii_lowercase, skip = False):
    l2 = len(s2)
    l = len(charset)
    res = ""
    pointer = 0
    
    for each in s1:
        this_key = s2[pointer]
        if each not in charset:
            if skip:
                pointer = (pointer + 1) % l2
            res += each
            continue
        i1 = charset.index(each)
        i2 = charset.index(this_key)
        res += charset[(i1 + i2) % l]
        pointer = (pointer + 1) % l2
    return res

def strsub(s1, s2, charset = string.ascii_lowercase, skip = False):
    l2 = len(s2)
    l = len(charset)
    res = ""
    pointer = 0
    
    for each in s1:
        this_key = s2[pointer]
        if each not in charset:
            if skip:
                pointer = (pointer + 1) % l2
            res += each
            continue
        i1 = charset.index(each)
        i2 = charset.index(this_key)
        res += charset[(i1 - i2 + l) % l]
        pointer = (pointer + 1) % l2
    return res

def solve_classical(text, decode_function = strsub, min_length = 0, max_length = 100, key_char = string.ascii_letters):
    length = get_key_length(text, max_length, min_length)[1]
    key = break_key(text, length, decode_function, key_char)
    print("=============================")
    print("found key: %s" % key)
    print("plaintext is below:")
    print(decode_function(text, key))
    return key

def decrypt_backpack(enc, public_key, dimension = 1):
    ans = []
    M = []
    length = len(public_key)
    for i in range(length):
        thisline = [0] * i + [2] + [0] * (length - i - 1) + [public_key[i]]
        M.append(thisline)
    const = dimension
    lastline = [const] * length + [enc]
    anslist = range(0, dimension + 1)
    M.append(lastline)
    M = Matrix(M)
    M = M.LLL()
    for eachline in M:
        this = eachline.list()
        if this[-1] != 0:
            continue
        this = [(-_ + const) // 2 for _ in this][:-1]
        check = True
        for each in this:
            if each not in anslist:
                check = False
                break
        
        if check:
            ans.append(this)
    return ans

def high_message_known(message, c, e, n, unknown_bit_length = 200):
    PR.<x> = PolynomialRing(Zmod(n))
    f = (x + message) ^ e - c
    f = f.monic()
    ans = f.small_roots(X = 2 ^ unknown_bit_length)[0]
    m = ans + message
    assert pow(m, e, n) == c
    return m

# display matrix picture with 0 and X
def matrix_overview(BB, bound):
    for ii in range(BB.dimensions()[0]):
        a = ('%02d ' % ii)
        for jj in range(BB.dimensions()[1]):
            a += '0' if BB[ii,jj] == 0 else 'X'
            a += ' '
        if BB[ii, ii] >= bound:
            a += '~'
        print(a)

def coppersmith_howgrave_univariate(pol, modulus, beta, mm, tt, XX, debug = False):
    """
    Coppersmith revisited by Howgrave-Graham
    
    finds a solution if:
    * b|modulus, b >= modulus^beta , 0 < beta <= 1
    * |x| < XX
    """
    #
    # init
    #
    dd = pol.degree()
    nn = dd * mm + tt

    #
    # checks
    #
    if not 0 < beta <= 1:
        raise ValueError("beta should belongs in (0, 1]")

    if not pol.is_monic():
        raise ArithmeticError("Polynomial must be monic.")

    #
    # calculate bounds and display them
    #
    """
    * we want to find g(x) such that ||g(xX)|| <= b^m / sqrt(n)

    * we know LLL will give us a short vector v such that:
    ||v|| <= 2^((n - 1)/4) * det(L)^(1/n)

    * we will use that vector as a coefficient vector for our g(x)
    
    * so we want to satisfy:
    2^((n - 1)/4) * det(L)^(1/n) < N^(beta*m) / sqrt(n)
    
    so we can obtain ||v|| < N^(beta*m) / sqrt(n) <= b^m / sqrt(n)
    (it's important to use N because we might not know b)
    """
    if debug:
        # t optimized?
        print("\n# Optimized t?\n")
        print("we want X^(n-1) < N^(beta*m) so that each vector is helpful")
        cond1 = RR(XX^(nn-1))
        print("* X^(n-1) = ", cond1)
        cond2 = pow(modulus, beta*mm)
        print("* N^(beta*m) = ", cond2)
        print("* X^(n-1) < N^(beta*m) \n-> GOOD" if cond1 < cond2 else "* X^(n-1) >= N^(beta*m) \n-> NOT GOOD")
        
        # bound for X
        print("\n# X bound respected?\n")
        print("we want X <= N^(((2*beta*m)/(n-1)) - ((delta*m*(m+1))/(n*(n-1)))) / 2 = M")
        print("* X =", XX)
        cond2 = RR(modulus^(((2*beta*mm)/(nn-1)) - ((dd*mm*(mm+1))/(nn*(nn-1)))) / 2)
        print("* M =", cond2)
        print("* X <= M \n-> GOOD" if XX <= cond2 else "* X > M \n-> NOT GOOD")

        # solution possible?
        print("\n# Solutions possible?\n")
        detL = RR(modulus^(dd * mm * (mm + 1) / 2) * XX^(nn * (nn - 1) / 2))
        print("we can find a solution if 2^((n - 1)/4) * det(L)^(1/n) < N^(beta*m) / sqrt(n)")
        cond1 = RR(2^((nn - 1)/4) * detL^(1/nn))
        print("* 2^((n - 1)/4) * det(L)^(1/n) = ", cond1)
        cond2 = RR(modulus^(beta*mm) / sqrt(nn))
        print("* N^(beta*m) / sqrt(n) = ", cond2)
        print("* 2^((n - 1)/4) * det(L)^(1/n) < N^(beta*m) / sqrt(n) \n-> SOLUTION WILL BE FOUND" if cond1 < cond2 else "* 2^((n - 1)/4) * det(L)^(1/n) >= N^(beta*m) / sqroot(n) \n-> NO SOLUTIONS MIGHT BE FOUND (but we never know)")

        # warning about X
        print("\n# Note that no solutions will be found _for sure_ if you don't respect:\n* |root| < X \n* b >= modulus^beta\n")
    
    #
    # Coppersmith revisited algo for univariate
    #

    # change ring of pol and x
    polZ = pol.change_ring(ZZ)
    x = polZ.parent().gen()

    # compute polynomials
    gg = []
    for ii in range(mm):
        for jj in range(dd):
            gg.append((x * XX)**jj * modulus**(mm - ii) * polZ(x * XX)**ii)
    for ii in range(tt):
        gg.append((x * XX)**ii * polZ(x * XX)**mm)
    
    # construct lattice B
    BB = Matrix(ZZ, nn)

    for ii in range(nn):
        for jj in range(ii+1):
            BB[ii, jj] = gg[ii][jj]

    # display basis matrix
    if debug:
        matrix_overview(BB, modulus^mm)

    # LLL
    BB = BB.LLL()

    # transform shortest vector in polynomial    
    new_pol = 0
    for ii in range(nn):
        new_pol += x**ii * BB[0, ii] / XX**ii

    # factor polynomial
    potential_roots = new_pol.roots()
    if debug:
        print("potential roots:", potential_roots)

    # test roots
    roots = []
    for root in potential_roots:
        if root[0].is_integer():
            result = polZ(ZZ(root[0]))
            if gcd(modulus, result) >= modulus^beta:
                roots.append(ZZ(root[0]))

    # 
    if roots:
        return roots
    else:
        return potential_roots

def factor_high_known(known, n, unknown_bit_length = 200):
    px = known
    PR.<x> = PolynomialRing(Zmod(n))
    f = x + px
    beta = 0.5
    dd = f.degree()
    epsilon = beta / 7
    mm = ceil(beta**2 / (dd * epsilon))
    tt = floor(dd * mm * ((1/beta) - 1))
    XX = 2 ^ unknown_bit_length
    roots = coppersmith_howgrave_univariate(f, n, beta, mm, tt, XX)
    p = roots[0][0] + known
    assert n % p == 0
    q = n // p
    return (p, q)


strict = False
helpful_only = True
dimension_min = 7

# display stats on helpful vectors
def helpful_vectors(BB, modulus):
    nothelpful = 0
    for ii in range(BB.dimensions()[0]):
        if BB[ii,ii] >= modulus:
            nothelpful += 1

    print(nothelpful, "/", BB.dimensions()[0], " vectors are not helpful")

# display matrix picture with 0 and X
def matrix_overview(BB, bound):
    for ii in range(BB.dimensions()[0]):
        a = ('%02d ' % ii)
        for jj in range(BB.dimensions()[1]):
            a += '0' if BB[ii,jj] == 0 else 'X'
            if BB.dimensions()[0] < 60:
                a += ' '
        if BB[ii, ii] >= bound:
            a += '~'
        print(a)

# tries to remove unhelpful vectors
# we start at current = n-1 (last vector)
def remove_unhelpful(BB, monomials, bound, current, debug = False):
    # end of our recursive function
    if current == -1 or BB.dimensions()[0] <= dimension_min:
        return BB

    # we start by checking from the end
    for ii in range(current, -1, -1):
        # if it is unhelpful:
        if BB[ii, ii] >= bound:
            affected_vectors = 0
            affected_vector_index = 0
            # let's check if it affects other vectors
            for jj in range(ii + 1, BB.dimensions()[0]):
                # if another vector is affected:
                # we increase the count
                if BB[jj, ii] != 0:
                    affected_vectors += 1
                    affected_vector_index = jj

            # level:0
            # if no other vectors end up affected
            # we remove it
            if affected_vectors == 0:
                if debug:
                    print("* removing unhelpful vector", ii)
                BB = BB.delete_columns([ii])
                BB = BB.delete_rows([ii])
                monomials.pop(ii)
                BB = remove_unhelpful(BB, monomials, bound, ii-1)
                return BB

            # level:1
            # if just one was affected we check
            # if it is affecting someone else
            elif affected_vectors == 1:
                affected_deeper = True
                for kk in range(affected_vector_index + 1, BB.dimensions()[0]):
                    # if it is affecting even one vector
                    # we give up on this one
                    if BB[kk, affected_vector_index] != 0:
                        affected_deeper = False
                # remove both it if no other vector was affected and
                # this helpful vector is not helpful enough
                # compared to our unhelpful one
                if affected_deeper and abs(bound - BB[affected_vector_index, affected_vector_index]) < abs(bound - BB[ii, ii]):
                    if debug:
                        print("* removing unhelpful vectors", ii, "and", affected_vector_index)
                    BB = BB.delete_columns([affected_vector_index, ii])
                    BB = BB.delete_rows([affected_vector_index, ii])
                    monomials.pop(affected_vector_index)
                    monomials.pop(ii)
                    BB = remove_unhelpful(BB, monomials, bound, ii-1)
                    return BB
    # nothing happened
    return BB

""" 
Returns:
* 0,0   if it fails
* -1,-1 if `strict=true`, and determinant doesn't bound
* x0,y0 the solutions of `pol`
"""
def boneh_durfee(pol, modulus, mm, tt, XX, YY, debug = False):
    """
    Boneh and Durfee revisited by Herrmann and May
    
    finds a solution if:
    * d < N^delta
    * |x| < e^delta
    * |y| < e^0.5
    whenever delta < 1 - sqrt(2)/2 ~ 0.292
    """

    # substitution (Herrman and May)
    PR.<u, x, y> = PolynomialRing(ZZ)
    Q = PR.quotient(x*y + 1 - u) # u = xy + 1
    polZ = Q(pol).lift()

    UU = XX*YY + 1

    # x-shifts
    gg = []
    for kk in range(mm + 1):
        for ii in range(mm - kk + 1):
            xshift = x^ii * modulus^(mm - kk) * polZ(u, x, y)^kk
            gg.append(xshift)
    gg.sort()

    # x-shifts list of monomials
    monomials = []
    for polynomial in gg:
        for monomial in polynomial.monomials():
            if monomial not in monomials:
                monomials.append(monomial)
    monomials.sort()
    
    # y-shifts (selected by Herrman and May)
    for jj in range(1, tt + 1):
        for kk in range(floor(mm/tt) * jj, mm + 1):
            yshift = y^jj * polZ(u, x, y)^kk * modulus^(mm - kk)
            yshift = Q(yshift).lift()
            gg.append(yshift) # substitution
    
    # y-shifts list of monomials
    for jj in range(1, tt + 1):
        for kk in range(floor(mm/tt) * jj, mm + 1):
            monomials.append(u^kk * y^jj)

    # construct lattice B
    nn = len(monomials)
    BB = Matrix(ZZ, nn)
    for ii in range(nn):
        BB[ii, 0] = gg[ii](0, 0, 0)
        for jj in range(1, ii + 1):
            if monomials[jj] in gg[ii].monomials():
                BB[ii, jj] = gg[ii].monomial_coefficient(monomials[jj]) * monomials[jj](UU,XX,YY)

    # Prototype to reduce the lattice
    if helpful_only:
        # automatically remove
        BB = remove_unhelpful(BB, monomials, modulus^mm, nn-1, debug)
        # reset dimension
        nn = BB.dimensions()[0]
        if nn == 0:
            print("failure")
            return 0,0

    # check if vectors are helpful
    if debug:
        helpful_vectors(BB, modulus^mm)
    
    # check if determinant is correctly bounded
    det = BB.det()
    bound = modulus^(mm*nn)
    if det >= bound:
        print("We do not have det < bound. Solutions might not be found.")
        print("Try with highers m and t.")
        if debug:
            diff = (log(det) - log(bound)) / log(2)
            print("size det(L) - size e^(m*n) = ", floor(diff))
        if strict:
            return -1, -1
    else:
        if debug:
            print("det(L) < e^(m*n) (good! If a solution exists < N^delta, it will be found)")

    # display the lattice basis
    if debug:
        matrix_overview(BB, modulus^mm)

    # LLL
    if debug:
        print("optimizing basis of the lattice via LLL, this can take a long time")

    BB = BB.LLL()

    if debug:
        print("LLL is done!")

    # transform vector i & j -> polynomials 1 & 2
    if debug:
        print("looking for independent vectors in the lattice")
    found_polynomials = False
    
    for pol1_idx in range(nn - 1):
        for pol2_idx in range(pol1_idx + 1, nn):
            # for i and j, create the two polynomials
            PR.<w,z> = PolynomialRing(ZZ)
            pol1 = pol2 = 0
            for jj in range(nn):
                pol1 += monomials[jj](w*z+1,w,z) * BB[pol1_idx, jj] / monomials[jj](UU,XX,YY)
                pol2 += monomials[jj](w*z+1,w,z) * BB[pol2_idx, jj] / monomials[jj](UU,XX,YY)

            # resultant
            PR.<q> = PolynomialRing(ZZ)
            rr = pol1.resultant(pol2)

            # are these good polynomials?
            if rr.is_zero() or rr.monomials() == [1]:
                continue
            else:
                if debug:
                    print("found them, using vectors", pol1_idx, "and", pol2_idx)
                found_polynomials = True
                break
        if found_polynomials:
            break

    if not found_polynomials:
        print("no independant vectors could be found. This should very rarely happen...")
        return 0, 0
    
    rr = rr(q, q)

    # solutions
    soly = rr.roots()

    if len(soly) == 0:
        print("Your prediction (delta) is too small")
        return 0, 0

    soly = soly[0][0]
    ss = pol1(q, soly)
    solx = ss.roots()[0][0]

    #
    return solx, soly

def solve_boneh_dufree(N, e, delta = .18, m = 4):
    t = int((1-2*delta) * m)
    X = 2*floor(N^delta)
    Y = floor(N^(1/2))
    P.<x,y> = PolynomialRing(ZZ)
    A = int((N+1)/2)
    pol = 1 + x * (A + y)
    solx, soly = boneh_durfee(pol, e, m, t, X, Y)
    d = int(pol(solx, soly) / e)
    return d

#debug()

'''
data1
Out[25]: b'\xd11\xdd\x02\xc5\xe6\xee\xc4i=\x9a\x06\x98\xaf\xf9\\/\xca\xb5\x87\x12F~\xab@\x04X>\xb8\xfb\x7f\x89U\xad4\x06\t\xf4\xb3\x02\x83\xe4\x88\x83%qAZ\x08Q%\xe8\xf7\xcd\xc9\x9f\xd9\x1d\xbd\xf2\x807<[\xd8\x82>1V4\x8f[\xaem\xac\xd46\xc9\x19\xc6\xddS\xe2\xb4\x87\xda\x03\xfd\x029c\x06\xd2H\xcd\xa0\xe9\x9f3B\x0fW~\xe8\xceT\xb6p\x80\xa8\r\x1e\xc6\x98!\xbc\xb6\xa8\x83\x93\x96\xf9e+o\xf7*p'

data2
Out[26]: b'\xd11\xdd\x02\xc5\xe6\xee\xc4i=\x9a\x06\x98\xaf\xf9\\/\xca\xb5\x07\x12F~\xab@\x04X>\xb8\xfb\x7f\x89U\xad4\x06\t\xf4\xb3\x02\x83\xe4\x88\x83%\xf1AZ\x08Q%\xe8\xf7\xcd\xc9\x9f\xd9\x1d\xbdr\x807<[\xd8\x82>1V4\x8f[\xaem\xac\xd46\xc9\x19\xc6\xddS\xe24\x87\xda\x03\xfd\x029c\x06\xd2H\xcd\xa0\xe9\x9f3B\x0fW~\xe8\xceT\xb6p\x80(\r\x1e\xc6\x98!\xbc\xb6\xa8\x83\x93\x96\xf9e\xabo\xf7*p'
'''