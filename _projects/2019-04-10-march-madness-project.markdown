---
layout: project
title:  "NCAA March Madness"
date:   2019-04-10 20:00:00 -0400
proj_num: 2
categories: project sports data_analysis
---

I participated in the Kaggle 2019 March Madness Prediction comptetion. The goal of competition is to predict probabilities of march madness and have the smallest log-loss. We used a logistic regression with Ken Pomeroy's adjusted offensive and defensive efficiency along with the teams free throw percentage as the variables. This model itself wasn't anything fancy, but we did have a few key improvements. The first was that we made full use of the two kaggle submissions by manually overriding on the first round game's probabilities to 100% and 0% and in the other submission the other way around. This guaranteed that we would get one game completely correct in our best submission lowering our log loss. Check out the code on my github [repo][march-madness-repo]

[march-madness-repo]: https://github.com/urolyi1/NCAA2019


