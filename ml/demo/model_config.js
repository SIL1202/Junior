// Auto-generated ML Model Config
const MODEL_CONFIG = {
  features: ['SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService', 'PaperlessBilling', 'MonthlyCharges', 'TotalCharges', 'ChargesRatio', 'IsNewCustomer', 'MonthToMonth_NewCustomer', 'gender_Male', 'MultipleLines_No phone service', 'MultipleLines_Yes', 'InternetService_Fiber optic', 'InternetService_No', 'OnlineSecurity_No internet service', 'OnlineSecurity_Yes', 'OnlineBackup_No internet service', 'OnlineBackup_Yes', 'DeviceProtection_No internet service', 'DeviceProtection_Yes', 'TechSupport_No internet service', 'TechSupport_Yes', 'StreamingTV_No internet service', 'StreamingTV_Yes', 'StreamingMovies_No internet service', 'StreamingMovies_Yes', 'Contract_One year', 'Contract_Two year', 'PaymentMethod_Credit card (automatic)', 'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check'],
  coefficients: [0.15534063330381107, 0.06492134996533977, -0.20366301039846418, -0.35127445238750793, -0.3310157578604407, 0.3439383154921069, -0.43924795040757464, -0.1077918959469219, 0.37786478239467536, -0.17507058128188127, 0.45395117456818657, 0.02786279529708128, -0.17706696277758377, 0.43208788886573873, 1.3071339983647314, -0.18062149540429434, -0.18062149540429434, -0.2641775445524245, -0.18062149540429434, -0.039556284902374436, -0.18062149540429434, 0.09367997996701528, -0.18062149540429434, -0.18099768562479251, -0.18062149540429434, 0.4679010762735432, -0.18062149540429434, 0.4811313732367006, -0.6690134698393257, -1.5536915002627172, 0.02076199200674446, 0.3675410707556307, -0.030210745153138503],
  intercept: -0.9471944325008415,
  scaler: {
    mean: {'tenure': 32.48509052183174, 'MonthlyCharges': 64.92996095136671, 'TotalCharges': 2299.33468228612, 'ChargesRatio': 0.15599230214441898},
    scale: {'tenure': 24.56656301538815, 'MonthlyCharges': 30.135430576006627, 'TotalCharges': 2279.001996496138, 'ChargesRatio': 0.280422176054764}
  },
  threshold: 0.45
};
