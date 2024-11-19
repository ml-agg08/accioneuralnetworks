# -*- coding: utf-8 -*-
"""Untitled15.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1zO8QkwgEVvgMYF2CAFaT9K9F2hgdRpkR
"""

import torch #for creating tensor objects, backprop etc
import torch.nn.functional as F #for one_hot

word = [
    "aarav", "vivaan", "aditya", "vihaan", "ishaan", "arjun", "sai", "ayaan", "krishna", "rohan",
    "reyansh", "shiv", "mohit", "karthik", "lakshman", "siddharth", "manish", "vishal", "vivek", "rahul",
    "manav", "nikhil", "kunal", "anish", "yash", "abhinav", "pranav", "amit", "gaurav", "ravi",
    "raj", "surya", "harsh", "tushar", "akash", "parth", "raghav", "ramesh", "ankit", "suresh",
    "pritam", "prem", "himesh", "ashwin", "dhruv", "sandeep", "vikas", "ajay", "anil", "madhav",
    "deepak", "bhuvan", "ashok", "ram", "shivansh", "nitin", "saurabh", "udit", "shaurya", "manoj",
    "chirag", "kiran", "amitabh", "nashit", "siddhi", "ritika", "ananya", "priya", "aishwarya", "sneha",
    "radhika", "meera", "swati", "pooja", "shruti", "simran", "nisha", "sanya", "kavya", "madhuri",
    "neha", "jaya", "mitali", "sonali", "laxmi", "vidya", "komal", "shalini", "tanu", "shreya",
    "nupur", "isha", "rupa", "divya", "ritu", "vandana", "pragya", "suman", "deepika", "manju",
    "shweta", "vaishnavi", "parul", "gayatri", "aarti", "tanvi", "chhavi", "anju", "tanisha", "sakshi",
    "simran", "sonal", "ravina", "meenal", "aarti", "shivani", "ankita", "kiran", "yamini", "sonalika"
]

#A list of strings; chatgpt generated these kerala origin names to train the model

import string
chars=list(string.ascii_lowercase)
stoi={s:i+1 for i,s in enumerate(chars)}
stoi['.']=0
itos={i+1:s for i,s in enumerate(chars)}
itos[0]="."

#since indexing is based on integers from 0 to 1,2,3.... we map each alphabet to an integer in that order itself

x=[]
xi=[]
y=[]
yi=[]
for w in word:
  w='.'+w+'.'
  for i in zip(w,w[1:]):
    x.append(i[0])
    xi.append(stoi[i[0]])
    y.append(i[1])
    yi.append(stoi[i[1]])

#using zip we create bigraphs and thus generating i/p and o/p labels

xi=torch.tensor(xi)
yi=torch.tensor(yi)
#creating tensor objects

xenc = F.one_hot(xi, num_classes=27).float() #for more efficient working of model, we use one_hat for classes

W = torch.randn((27, 27),requires_grad=True) #since its a single layer nn with 27 neruons we need a 27x27 weight matrix

#forward pass matrix multiplication (we use @ for product)
logits=xenc@W
counts=logits.exp()
probs=counts/counts.sum(dim=1,keepdim=True)
probs.shape

#here we consider cross entropy loss, basically sum of negative logs of each predicted probability values corresponding to actual labels (we already have them for training set)
nlls=torch.zeros(len(yi))
j=0
for i in yi:
  loss=torch.log(probs[j,i])
  nlls[j]=-loss
  j+=1

print('=========')
print('average negative log likelihood, i.e. loss =', nlls.mean().item())
LOSS=nlls.mean()

W.grad=None
LOSS.backward()
#backprop to calculate gradients of each weights like how they affect the final loss

#conducting various epochs and using manual gradient descent to converge to min loss and get corresponding wieghts
alpha=1
for i in range(50):
  W.data+=-alpha*W.grad

  #forward pass
  logits=xenc@W
  counts=logits.exp()
  probs=counts/counts.sum(dim=1,keepdim=True)
  probs.shape

  nlls=torch.zeros(len(yi))
  j=0
  for i in yi:
    loss=torch.log(probs[j,i])
    nlls[j]=-loss
    j+=1

  print('=========')
  print('average negative log likelihood, i.e. loss =', nlls.mean().item()+0.01*(W**2).mean())
  LOSS=nlls.mean()+0.01*(W**2).mean()

g = torch.Generator().manual_seed(2147483647)
for i in range(30):
  out =[]
  ix=0
  while True:
    inputencode=F.one_hot(torch.tensor([ix]),num_classes=27).float()
    logits=inputencode@W
    counts=logits.exp()
    p=counts/counts.sum(1,keepdims=True)

    ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
    out.append(itos[ix])
    if ix == 0:
      break
  print(out)

  #testing yayyy !!
