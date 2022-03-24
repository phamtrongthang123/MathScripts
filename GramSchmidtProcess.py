import numpy as np
from scipy.integrate import quad
 
def polynomial(x, coeffs):
    """compute polynomial of degree len(coeffs)

    Args:
        x (float): input x
        coeffs (list[float]): list of coeffs [a,b,c...] with a + bx + cx^2 + ...

    Returns:
        float: value of the function
    """
    res = 0
    for i,c in enumerate(coeffs):
        res += x**i * c
    return res 
 
def inner_product(coeffs_f, coeffs_g):
    """compute inner product. You can modify this function to change it into other inner product.
    Currently this function will compute <f,g> = int_0^1 (f(x)g(x))dx.

    Args:
        coeffs_f (list[float]): list of coefficients, this represents for the f polynomial function
        coeffs_g (list[float]): same as coeffs_f, but for g function

    Returns:
        float: value of the inner product
    """
    res = quad(lambda x: polynomial(x,coeffs_f)*polynomial(x,coeffs_g), 0,1)
    return res[0]
 
def generate_linearly_independent(n):
    """This function generates all linearly independent vectors with length = n following the rule: 
       - v_i = (...,0,1,0,0,...) with 1 stays at i-th position, and i from 0 to n-1. 

    Args:
        n (int): the degree / length n

    Returns:
        list[list[int]]: list of vectors that are independent to each other.
    """
    res = []
    for i in range(n):
        tmp = [0] * n 
        tmp[i] = 1
        res.append(tmp)
    return res 
 
def proj(u,v):
    """Projection function

    Args:
        u (list[float]): first vector.  
        v (list[float]): second vector.

    Returns:
        list[float]: the projected vector from u to v.
    """
    return inner_product(u,v) / inner_product(u,u) * u 
 
def gram_schmidt(n):
    """Gram-Schmidth process. Follow: https://en.wikipedia.org/wiki/Gram%E2%80%93Schmidt_process

    Args:
        n (int): degree of the polynomial functions

    Returns:
        list[list[float]]: the orthonormal basis
    """ 
    linearly_independent_set = np.array(generate_linearly_independent(n)).astype('float64')
    ulist = [] 
    for k in range(n):
        if k == 0:
            ulist.append(linearly_independent_set[k])
            continue
        uk = linearly_independent_set[k]
        for j in range(k):
            uk -= proj(ulist[j], linearly_independent_set[k])
        ulist.append(uk)
    elist = [] 
    for uk in ulist:
        elist.append(uk/np.sqrt(inner_product(uk,uk)))
    return elist
 

 
def check_output(elist, eps = 1e-2):
    """Sanity check.

    Args:
        elist (list[list[float]]): the orthonormal basis.
        eps (float, optional): epsilon for float comparation. Defaults to 1e-2.
    """
    for ek in elist: 
        norm = np.sqrt(inner_product(ek,ek))
        if np.abs(norm - 1) < eps:
            continue
        print("wrong norm", norm)
    for i in range(len(elist)):
        for j in range(len(elist)):
            inner = inner_product(elist[i], elist[j])
            if i == j and np.abs(inner - 1) < eps :
                continue
            elif i != j and inner < eps:
                continue
            print("wrong", i, j, inner)
 
if __name__=="__main__":
    n = 10
    elist = gram_schmidt(n+1)
    print(elist)
    check_output(elist)