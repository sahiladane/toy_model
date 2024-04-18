'''
So, the example has two components/dimensions. One - the state of the world that people have a prior on, and the reputation of Fox news, that affects the probability of signal that we observe. 

states = {L, R}
reputation_state_of_Fox = {0 - Fox lies, 1 - Fox is truthful}
signals  = {l, r}

Alice and Bob both agree on the experiment of course. If Fox is truthful, they agree that l is 'native' to L. That is, given state L and Fox is truthful, Fox will send out l more often than r. Conversely though, if Fox lies, they agree that r is 'native' to L. That is, given state L and Fox lies, Fox will send out r more often than l. 

Read left_given_truthful_left as probability that Fox sends out left given that Fox is truthful and the state is actually left. 

Prior is defined as the probabilty that the person thinks the state is R. 

Think of Alice as a liberal person. She thinks that probability that Fox is truthful is 0.4 < 0.6, that of Bob's. You can use the same example to show two things,

1) If Alice and Bob start out at 0.4, 0.6, you can see their posteriors for these numbers. They diverge. 0.63 - 0.36 > 0.6 - 0.4

2) If Alice and both Bob start out at 0.5, they update in different directions. 

In my example, left_given_truthful_left + left_given_truthful_right sum to 1. But they obviously need not.

I also update the prior of Alice and Bob on how truthful Fox is and they still keep diverging


'''

import random as r

prior_alice = 0.5
prior_bob = 0.5

fox_truthful_alice = 0.4
fox_truthful_bob = 0.6

def bayes_updating_l(prior_alice, prior_bob, fox_truthful_alice, fox_truthful_bob):

        left_given_truthful_left = 0.7
        left_given_truthful_right = 0.3

        left_given_lies_left = 0.3
        left_given_lies_right = 0.7

        # posterior after observing left 

        posterior_alice_l = prior_alice * ( (1 - fox_truthful_alice) * left_given_lies_right + fox_truthful_alice * left_given_truthful_right)/ ( prior_alice * ( (1 - fox_truthful_alice) * left_given_lies_right + fox_truthful_alice * left_given_truthful_right) + (1-prior_alice) * ( (1 - fox_truthful_alice) * left_given_lies_left + fox_truthful_alice * left_given_truthful_left) )

        #print(posterior_alice_l)

        posterior_bob_l = prior_bob * ( (1 - fox_truthful_bob) * left_given_lies_right + fox_truthful_bob * left_given_truthful_right)/ ( prior_bob * ( (1 - fox_truthful_bob) * left_given_lies_right + fox_truthful_bob * left_given_truthful_right) + (1-prior_bob) * ( (1 - fox_truthful_bob) * left_given_lies_left + fox_truthful_bob * left_given_truthful_left) )

        #print(posterior_bob_l)

        fox_posterior_alice_l = fox_truthful_alice * ( prior_alice * left_given_truthful_right + (1-prior_alice) * left_given_truthful_left) / (fox_truthful_alice * ( prior_alice * left_given_truthful_right + (1-prior_alice) * left_given_truthful_left) + (1-fox_truthful_alice) * ( prior_alice * left_given_lies_right + (1-prior_alice) * left_given_lies_left) )

        #print(fox_posterior_alice_l)

        fox_posterior_bob_l = fox_truthful_bob * ( prior_bob * left_given_truthful_right + (1-prior_bob) * left_given_truthful_left) / (fox_truthful_bob * ( prior_bob * left_given_truthful_right + (1-prior_bob) * left_given_truthful_left) + (1-fox_truthful_bob) * ( prior_bob * left_given_lies_right + (1-prior_bob) * left_given_lies_left) )

        #print(fox_posterior_bob_l,'\n')

        return posterior_alice_l, posterior_bob_l, fox_posterior_alice_l, fox_posterior_bob_l

def bayes_updating_r(prior_alice, prior_bob, fox_truthful_alice, fox_truthful_bob):

        left_given_truthful_left = 0.7
        left_given_truthful_right = 0.3

        left_given_lies_left = 0.3
        left_given_lies_right = 0.7

        # posteriors after observing right

        posterior_alice_r = prior_alice * ((1 - fox_truthful_alice) * (1-left_given_lies_right) + fox_truthful_alice * (1-left_given_truthful_right))/ ( prior_alice * ( (1 - fox_truthful_alice) * (1-left_given_lies_right) + fox_truthful_alice * (1-left_given_truthful_right)) + (1-prior_alice) * ( (1 - fox_truthful_alice) * (1-left_given_lies_left) + fox_truthful_alice * (1-left_given_truthful_left)) )

        #print(posterior_alice_r)

        posterior_bob_r = prior_bob * ( (1 - fox_truthful_bob) * (1-left_given_lies_right) + fox_truthful_bob * (1-left_given_truthful_right))/ ( prior_bob * ( (1 - fox_truthful_bob) * (1-left_given_lies_right) + fox_truthful_bob * (1-left_given_truthful_right)) + (1-prior_bob) * ( (1 - fox_truthful_bob) * (1-left_given_lies_left) + fox_truthful_bob * (1-left_given_truthful_left)) )

        #print(posterior_bob_r)

        fox_posterior_alice_r = fox_truthful_alice * ( prior_alice * (1-left_given_truthful_right) + (1-prior_alice) * (1-left_given_truthful_left)) / (fox_truthful_alice * ( prior_alice * (1-left_given_truthful_right) + (1-prior_alice) * (1-left_given_truthful_left)) + (1-fox_truthful_alice) * ( prior_alice * (1-left_given_lies_right) + (1-prior_alice) * (1-left_given_lies_left)) )

        #print(fox_posterior_alice_r)

        fox_posterior_bob_r = fox_truthful_bob * ( prior_bob * (1-left_given_truthful_right) + (1-prior_bob) * (1-left_given_truthful_left)) / (fox_truthful_bob * ( prior_bob * (1-left_given_truthful_right) + (1-prior_bob) * (1-left_given_truthful_left)) + (1-fox_truthful_bob) * ( prior_bob * (1-left_given_lies_right) + (1-prior_bob) * (1-left_given_lies_left)) )

        #print(fox_posterior_bob_r,'\n')

        return posterior_alice_r, posterior_bob_r, fox_posterior_alice_r, fox_posterior_bob_r


for i in range(10000):
        # p= probability of observing r
        ex = r.binomialvariate(n=1,p=0.7)
        if ex == 1:
                a, b, c, d = bayes_updating_r(prior_alice, prior_bob, fox_truthful_alice, fox_truthful_bob)
                prior_alice, prior_bob, fox_truthful_alice, fox_truthful_bob = a, b, c, d
        else:
                a, b, c, d = bayes_updating_l(prior_alice, prior_bob, fox_truthful_alice, fox_truthful_bob)
                prior_alice, prior_bob, fox_truthful_alice, fox_truthful_bob = a, b, c, d

print(prior_alice)
print(fox_truthful_alice)
print(prior_bob)
print(fox_truthful_bob)



