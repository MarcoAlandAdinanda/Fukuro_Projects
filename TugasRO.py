import math

def po_formula():
    lamda = 7
    mu = 1.25
    s = 10

    denominator = 0
    for n in range(s): 
        denominator += (((lamda/mu)**n) / (math.factorial(n))) 
        
    return 1/(denominator + ((((lamda/mu)**s) / (math.factorial(s))) * (1/(1-(lamda/(s*mu))))))

if __name__ == "__main__":
    print(f"{po_formula():5f}")



    