# Belgian Load Forecasting

## Plan of action
- [ ] Models:
    - [ ] Linear Model
    - [ ] Ridge Regression
    - [ ] SVM
    - [ ] CNN
    - [ ] LightGBM
    - [ ] Linear/Non-Linear NN
- [ ] For every question/model
    - [ ] Conformal prediction

- [ ] Very short time forecasting (VST - 1 hour ahead)
    - [ ] What is the optimal input window size ?
    - [ ] Do additional input features help ?
    - [ ] Do recursive forecasting help ?
    - [ ] Which concrete applications can benefit from such short forecasting ?
    - [ ] Is local linearity extendable to Britain ?
- [ ] Short time forecasting (ST - 30 hours ahead)
    - [ ] What is the optimal input window size ?
    - [ ] Do additional input features help ?
    - [ ] Do recursive forecasting help ?
    - [ ] Do the VST forecasting results hold ?
    - [ ] How does undersampling time link ST to VST ?
    - [ ] Is the length of the output window defining the quality of the results, or is it the time horizon (is predicting the 0h->30h equivalent to predict 29h->30h)
    - [ ] Do the first forecasted steps compete with VST ?
- [ ] Other
    - [ ] At which forecasting horizon the VST best methods are not better than ELIA ?
    - [ ] Are the methods for VST and ST combinable for intermediary horizon ?