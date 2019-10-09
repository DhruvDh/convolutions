# Homework 1

- Dhruv Dhamani
- ddhamani@uncc.edu

## Table of Contents

- [Homework 1](#homework-1)
  - [Table of Contents](#table-of-contents)
  - [Question 1](#question-1)
  - [Question 2](#question-2)
  - [Question 3](#question-3)
  - [Question 4](#question-4)
  - [Question 5](#question-5)

## Question 1

- 223.1.17.0/26
- 223.1.17.128/25
- 223.1.17.192/28

## Question 2

> Part a
- Subnet `A` `214.97.255/24` - `256` addresses
- Subnet `B` `214.97.254.0/25` - `214.97.254.0/29` - `120` addresses
- Subnet `C` `214.97.254.128/25` - `128` addresses
- Subnet `D` `214.97.254.0/31` - `2` addresses
- Subnet `E` `214.97.254.2/31` - `2` addresses
- Subnet `F` `214.97.254.4/30` - `4` addressess

> Part b

- Router 1

| Longest Prefix Match | Outgoing Interface |
| --- | --- |
| `11010110 01100001 11111111` | Subnet `A` |
| `11010110 01100001 11111110 0000000` | Subnet `D` |
| `11010110 01100001 11111110 000001` | Subnet `F` |

- Router 2

| Longest Prefix Match | Outgoing Interface |
| --- | --- |
| `11010110 01100001 11111111 0000000` | Subnet `D` |
| `11010110 01100001 11111110 0` | Subnet `B` |
| `11010110 01100001 11111110 0000001` | Subnet `E` |

- Router 3

| Longest Prefix Match | Outgoing Interface |
| --- | --- |
| `11010110 01100001 11111111 000001` | Subnet `F` |
| `11010110 01100001 11111110 0000001` | Subnet `E` |
| `11010110 01100001 11111110 1` | Subnet `C` |


## Question 3

Cost matrix -

|1. | u | v | x | y | z |
|---|---|---|---|---|---|
| v | ∞ | ∞ | ∞ | ∞ | ∞ |
| x | ∞ | ∞ | ∞ | ∞ | ∞ |
| z | ∞ | 6 | 2 | ∞ | 0 |


| 2. | u | v | x | y | z |
|---|---|---|---|---|---|
| v | 1 | 0 | 3 | ∞ | 6 |
| x | ∞ | 3 | 0 | 3 | 2 |
| z | 7 | 5 | 2 | 5 | 0 |

| 3.| u | v | x | y | z |
|---|---|---|---|---|---|
| v | 1 | 0 | 3 | 3 | 5 |
| x | 4 | 3 | 0 | 3 | 2 |
| z | 6 | 5 | 2 | 5 | 0 |

| 4.| u | v | x | y | z |
|---|---|---|---|---|---|
| v | 1 | 0 | 3 | 3 | 5 |
| x | 4 | 3 | 0 | 3 | 2 |
| z | 6 | 5 | 2 | 5 | 0 |


## Question 4

| $Step$ | $N$ | $D(s),p(s)$ | $D(t),p(t)$ | $D(v),p(v)$ | $D(w),p(w)$ | $D(x),p(x)$ | $D(y),p(y)$ | $D(z),p(z)$ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | u | ∞ | ∞ | 3,u | 2,u | ∞| ∞| ∞|
| 1 | uw | 6,w | ∞| 3,u | 2,u | 3,w | ∞| ∞|
| 2 | uwv | 6,w | ∞| 3,u | 2,u | 3,w | 5,v | ∞|
| 3 | uwvx | 6,w | 8,x | 3,u | 2,u | 3,w | 5,v | 7,x |
| 4 | uwvxy | 6,w | 8,x | 3,u | 2,u | 3,w | 5,v | 6,y |
| 5 | uwvxys | 6,w | 8,x | 3,u | 2,u | 3,w | 5,v | 6,y |
| 6 | uwvxysz | 6,w | 8,x | 3,u | 2,u | 3,w | 5,v | 6,y |
| 7 | uwvxyst | 6,w | 8,x | 3,u | 2,u | 3,w | 5,v | 6,y |

## Question 5

> Part a.

| $Step$ | $N$ | $D(x),p(x)$ | $D(u),p(u)$ | $D(v),p(v)$ | $D(w),p(w)$ | $D(y),p(y)$ | $D(z),p(z)$ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | t | ∞| 2,t | 4,t | ∞| 7,t | ∞|
| 1 | tu | ∞| 2,t | 4,t | 5,u | 7,t | ∞|
| 2 | tuv | 7,v | 2,t | 4,t | 5,u | 7,t | ∞|
| 3 | tuvw | 7,v | 2,t | 4,t | 5,u | 7,t | ∞|
| 4 | tuvwx | 7,v | 2,t | 4,t | 5,u | 7,t | 15,x |
| 5 | tuvwxy | 7,v | 2,t | 4,t | 5,u | 7,t | 15,x |
| 6 | tuvwxyz | 7,v | 2,t | 4,t | 5,u | 7,t | 15,x |

> Part b.

| $Step$ | $N$ | $D(x),p(x)$ | $D(u),p(u)$ | $D(v),p(v)$ | $D(w),p(w)$ | $D(y),p(y)$ | $D(z),p(z)$ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | u | ∞| 2,u | 3,u | 3,u | ∞| ∞|
| 1 | ut | ∞| 2,u | 3,u | 3,u | 9,t | ∞|
| 2 | utv | 6,v | 2,u | 3,u | 3,u | 9,t | ∞|
| 3 | utvw | 6,v | 2,u | 3,u | 3,u | 9,t | ∞|
| 4 | utvwx | 6,v | 2,u | 3,u | 3,u | 9,t | 14,x |
| 5 | utvwxy | 6,v | 2,u | 3,u | 3,u | 9,t | 14,x |
| 6 | utvwxyz | 6,v | 2,u | 3,u | 3,u | 9,t | 14,x |

> Part c.

| $Step$| $N$ | $D(x),p(x)$ | $D(u),p(u)$ | $D(v),p(v)$ | $D(w),p(w)$ | $D(y),p(y)$ | $D(z),p(z)$ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | v | 3,v | 3,v | 4,v | 4,v | 8,v | ∞|
| 1 | vx | 3,v | 3,v | 4,v | 4,v | 8,v | 11,x |
| 2 | vxu | 3,v | 3,v | 4,v | 4,v | 8,v | 11,x |
| 3 | vxut | 3,v | 3,v | 4,v | 4,v | 8,v | 11,x |
| 4 | vxutw | 3,v | 3,v | 4,v | 4,v | 8,v | 11,x |
| 5 | vxutwy | 3,v | 3,v | 4,v | 4,v | 8,v | 11,x |
| 6 | vxutwyz | 3,v | 3,v | 4,v | 4,v | 8,v | 11,x |

> Part d.

| $Step$| $N$ | $D(x),p(x)$ | $D(u),p(u)$ | $D(v),p(v)$ | $D(w),p(w)$ | $D(y),p(y)$ | $D(z),p(z)$ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | w | 6,w | 3,w | 4,w | ∞| ∞| ∞|
| 1 | wu | 6,w | 3,w | 4,w | 5,u | ∞| ∞|
| 2 | wuv | 6,w | 3,w | 4,w | 5,u | 12,v | ∞|
| 3 | wuvt | 6,w | 3,w | 4,w | 5,u | 12,v | ∞|
| 4 | wuvtx | 6,w | 3,w | 4,w | 5,u | 12,v | 14,x |
| 5 | wuvtxy | 6,w | 3,w | 4,w | 5,u | 12,v | 14,x |
| 6 | wuvtxyz | 6,w | 3,w | 4,w | 5,u | 12,v | 14,x |

> Part e.

| $Step$| $N$ | $D(x),p(x)$ | $D(u),p(u)$ | $D(v),p(v)$ | $D(w),p(w)$ | $D(y),p(y)$ | $D(z),p(z)$ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | y | 6,y | ∞| 8,y | ∞| 7,y | 12,y |
| 1 | yx | 6,y | ∞| 8,y | 12,x | 7,y | 12,y |
| 2 | yxt | 6,y | 9,t | 8,y | 12,x | 7,y | 12,y |
| 3 | yxtv | 6,y | 9,t | 8,y | 12,x | 7,y | 12,y |
| 4 | yxtvu | 6,y | 9,t | 8,y | 12,x | 7,y | 12,y |
| 5 | yxtvuw | 6,y | 9,t | 8,y | 12,x | 7,y | 12,y |
| 6 | yxtvuwz | 6,y | 9,t | 8,y | 12,x | 7,y | 12,y |

> Part f.
 
| $Step$  | $N$ | $D(x),p(x)$ | $D(u),p(u)$ | $D(v),p(v)$ | $D(w),p(w)$ | $D(y),p(y)$ | $D(z),p(z)$ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | z | 8,z | ∞| ∞| ∞| 12,z | ∞|
| 1 | zx | 8,z | ∞| 11,x | 14,x | 12,z | ∞|
| 2 | zxv | 8,z | 14,v | 11,x | 14,x | 12,z | 15,v |
| 3 | zxvy | 8,z | 14,v | 11,x | 14,x | 12,z | 15,v |
| 4 | zxvyu | 8,z | 14,v | 11,x | 14,x | 12,z | 15,v |
| 5 | zxvyuw | 8,z | 14,v | 11,x | 14,x | 12,z | 15,v |
| 6 | zxvyuwt | 8,z | 14,v | 11,x | 14,x | 12,z | 15,v |