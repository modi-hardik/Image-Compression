from numpy.core.fromnumeric import repeat
from fft_ifft import *
from rsa import *
from generateNbitPrimeNumber import *
import numpy as np
import cv2 as cv
from matplotlib.image import imread
import matplotlib.pyplot as plt
from PIL import Image
import timeit

print("\n\n\n\n------------------------------------------------------------------------------------\n")
print("---------------------------------1D-FFT and 1D-DFFT--------------------------------\n")
print("------------------------------------------------------------------------------------\n")

x = np.random.random(1024)
y = np.random.random(1024)

#Checking the results of FFT and DFT
print("Comparing results of FFT and DFT : ")
print("A : ",np.allclose(FFT_1D(x), DFT_1D(x)))
print("B : ",np.allclose(FFT_1D(y), DFT_1D(y)))

x = np.pad(x, (0, 1024), 'constant')
y = np.pad(y, (0, 1024), 'constant')

SETUP_CODE = '''
import numpy as np
from fft_ifft import FFT_1D 
from fft_ifft import DFT_1D
'''
 
TEST_CODE = '''
x = np.random.random(1024)
FFT_1D(x)
'''
TEST_CODE1 = '''
x = np.random.random(1024)
DFT_1D(x)
'''

print("\n\n\n\n------------------------------------------------------------------------------------\n")
print("---------------------------------Time Comparisons--------------------------------\n")
print("------------------------------------------------------------------------------------\n")

print("Time taken by DFT : ",timeit.repeat(setup=SETUP_CODE,stmt=TEST_CODE1,repeat=1,number=1))
print("Time taken by FFT : ",timeit.repeat(setup=SETUP_CODE,stmt=TEST_CODE,repeat=1,number=1))

#Converting Coefficient form to P-V from 
va = FFT_1D(x)
vb = FFT_1D(y)
#Pointwise multiplication to produce c(x) in P-V form
c = np.multiply(va,vb)
#Converting P-V form to Coefficient form back
vc=inverseFFT_1D(c)
vc=np.asarray(vc, dtype=float) #Converting into array form


#con=np.convolve(x,y)
#con=np.asarray(con, dtype=float) #Converting into array form
t=np.polynomial.polynomial.polymul(x, y)
t=np.asarray(t, dtype=float) #Converting into array form

#Checking our functions fft and ifft
print("\n\nChecking fft:",np.allclose(va, np.fft.fft(x)))
print("Checking ifft:",np.allclose(vc, np.fft.ifft(c)))

print('\n\nX:',x,'\nY:',y)
print('\nX*Y (P-V):\n',c,'\n\n')
m=vc[:-1]
print('Ifft X*Y :\n',m)
print('Polynoial Multiplication X*Y:\n',t)
#print('Poly Convolve:\n',con)
print("\n\nChecking if both the values are same:",np.allclose(m,t))



print("\n\n\n\n------------------------------------------------------------------------------------\n")
print("---------------------------------2D-FFT and 2D-IDFFT--------------------------------\n")
print("------------------------------------------------------------------------------------\n")

x_2d=np.random.randint(255, size=(4,8))
va_2d = FFT_2D(x_2d)
vc_2d=inverseFFT_2D(va_2d)

#Checking our functions fft and ifft
print("Checking 2d-fft:",np.allclose(va_2d, np.fft.fft2(x_2d)))
print("Checking 2d-ifft:",np.allclose(vc_2d, np.fft.ifft2(va_2d)))

print('\n\nX:\n',x_2d)
print('2f-fft :\n',va_2d)
vc_2d=np.asarray(vc_2d, dtype=float) #Converting into array form
print('2d-Ifft :\n',vc_2d)

print("\n\nChecking if both the values are same:",np.allclose(vc_2d,x_2d))





#RSA encryption
print("\n\n\n\n------------------------------------------------------------------------------------\n")
print("-----------------------------------RSA encryption-----------------------------------\n")
print("------------------------------------------------------------------------------------\n")

p = 2
q = 2
while True:
        #n-bit number
		n = 256
		prime_candidate = getLowLevelPrime(n)
		if not isMillerRabinPassed(prime_candidate):
			continue
		else:
			p = prime_candidate
			break

while True:
        #n-bit number
		n = 256
		prime_candidate = getLowLevelPrime(n)
		if not isMillerRabinPassed(prime_candidate):
			continue
		else:
			q = prime_candidate
			break

print("p : ",p)
print("q : ",q)
public, private = generate_key_pair(p, q)
message = np.array_str(c)

print('\n\nPublic Key [{}, {}]'.format(public[0],len(message)))
#print('Private Key [{}, {}]'.format(private[0],len(message)))

print('\n\nInitial Message:',message,'\n')
encrypted_msg = encrypt(public,message)
decrypted_msg = decrypt(private,encrypted_msg)
ex = np.array(encrypted_msg)
cx = np.array(decrypted_msg)
#print('\n\nEncrypted Message:',ex,type(ex))
print('Decrypted Message:',cx,'\n\n')






print("\n\n\n\n------------------------------------------------------------------------------------\n")
print("---------------------------------Picture Reduction----------------------------------\n")
print("------------------------------------------------------------------------------------\n")
A=imread('earth.jpg')
plt.figure()
#plt.imshow(256-A)
plt.imshow(A[:,:,1], cmap='gray', vmin = 0, vmax = 255,interpolation='none')
plt.axis('off')
plt.savefig('GrayScale.png')
print('Gray Scaled Image is saved as "GrayScale.png" into the same directory')
#plt.show()
B=np.mean(A,-1) #converting into grayscale image
Bt=FFT_2D(B)
Btsort=np.sort(np.abs(Bt.reshape(-1)))
keep=0.1 #keeping top 10% values of the enlarged scale
thresh=Btsort[int(np.floor((1-keep)*len(Btsort)))] #keeping a threshold value for the keep value chosen
ind=np.abs(Bt)>thresh
Btlow=Bt*ind
Alow=inverseFFT_2D(Btlow).real
plt.figure()
plt.imshow(256-Alow,cmap='gray')
plt.axis('off')
plt.savefig('Compressed.png')
print('Compressed Image is saved as "Compressed.png" into the same directory')
#plt.show()
