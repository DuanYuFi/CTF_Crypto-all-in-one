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

class ECC(Point):
    
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



def debug():
    print(Pollard_rho(15))



#debug()

'''
data1
Out[25]: b'\xd11\xdd\x02\xc5\xe6\xee\xc4i=\x9a\x06\x98\xaf\xf9\\/\xca\xb5\x87\x12F~\xab@\x04X>\xb8\xfb\x7f\x89U\xad4\x06\t\xf4\xb3\x02\x83\xe4\x88\x83%qAZ\x08Q%\xe8\xf7\xcd\xc9\x9f\xd9\x1d\xbd\xf2\x807<[\xd8\x82>1V4\x8f[\xaem\xac\xd46\xc9\x19\xc6\xddS\xe2\xb4\x87\xda\x03\xfd\x029c\x06\xd2H\xcd\xa0\xe9\x9f3B\x0fW~\xe8\xceT\xb6p\x80\xa8\r\x1e\xc6\x98!\xbc\xb6\xa8\x83\x93\x96\xf9e+o\xf7*p'

data2
Out[26]: b'\xd11\xdd\x02\xc5\xe6\xee\xc4i=\x9a\x06\x98\xaf\xf9\\/\xca\xb5\x07\x12F~\xab@\x04X>\xb8\xfb\x7f\x89U\xad4\x06\t\xf4\xb3\x02\x83\xe4\x88\x83%\xf1AZ\x08Q%\xe8\xf7\xcd\xc9\x9f\xd9\x1d\xbdr\x807<[\xd8\x82>1V4\x8f[\xaem\xac\xd46\xc9\x19\xc6\xddS\xe24\x87\xda\x03\xfd\x029c\x06\xd2H\xcd\xa0\xe9\x9f3B\x0fW~\xe8\xceT\xb6p\x80(\r\x1e\xc6\x98!\xbc\xb6\xa8\x83\x93\x96\xf9e\xabo\xf7*p'
'''
