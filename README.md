# Optimization over Smooth Abstract Domains

This project is adapted from S. Chaudhuri, M. Clochard, and A. Solar-Lezama, “Bridging boolean and quantitative synthesis using smoothed proof search,” 2014.

Abstract interpretation allows us to track possible program states at given program points. However, the function from input states to output states is very discontinuous.
By creating a smoothing operator, we can optimize over unknowns while maintaining boolean properties.

This project is an implementation of the interval abstract domain with smoothing.

## Getting Started

This project uses Pytorch with Python 3.
