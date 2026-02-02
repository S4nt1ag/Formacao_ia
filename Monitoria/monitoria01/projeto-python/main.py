from modulos import soma, subtracao
from funcoes.multiplicacao import multiplica

a = 10
b = 5

som = soma(a, b)
sub = subtracao(a, b)
mult = multiplica(a, b)

print("Soma igual a " + str(som))
print("Subtracao igual a " + str(sub))
print("multiplicacao igual a " + str(mult))