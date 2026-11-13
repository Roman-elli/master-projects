import pandas as pd

def analyze_data(df):
    """Imprime métricas e distribuições para análise de sanidade dos dados."""
    print("\n--- Balanceamento da Classe Alvo (No-show) ---")
    target_counts = df['No-show'].value_counts()
    target_pct = df['No-show'].value_counts(normalize=True) * 100
    print(pd.DataFrame({'Contagem': target_counts, 'Percentagem (%)': target_pct.round(2)}))
        
    print("--- Contagem de Valores Nulos ---")
    print(df.isnull().sum())
    
    categorical_features = ['Gender', 'Scholarship', 'Hipertension', 'Diabetes', 'Alcoholism', 'Handcap', 'SMS_received', 'Neighbourhood']
    print("\n--- Distribuição das Variáveis Categóricas (%) ---")
    for col in categorical_features:
        if col in df.columns:
            print(f"\n{col}:")
            print((df[col].value_counts(normalize=True) * 100).round(2))

def simulate_operational_efficiency(y_true, y_pred, appointment_duration_mins=30, manual_call_mins=3, intervention_success_rate=0.40):
    """
    Simula o impacto na eficiência hospitalar (horas médicas e administrativas poupadas).
    """
    # 1. Totais
    total_patients = len(y_true)
    total_actual_noshows = sum(y_true)
    
    # 2. Matriz de Confusão do Modelo
    true_positives = sum((y_true == 1) & (y_pred == 1))   # Modelo previu falta, e ia faltar
    false_positives = sum((y_true == 0) & (y_pred == 1))  # Modelo previu falta, mas ia aparecer
    false_negatives = sum((y_true == 1) & (y_pred == 0))  # Modelo falhou em prever a falta
    
    # 3. IMPACTO ADMINISTRATIVO (Horas gastas a confirmar consultas)
    admin_hours_no_ai = (total_patients * manual_call_mins) / 60
    
    # Cenário com IA: A equipa (ou um sistema automático) só foca na "Lista de Alto Risco" (TP + FP)
    admin_hours_with_ai = ((true_positives + false_positives) * manual_call_mins) / 60
    admin_hours_saved = admin_hours_no_ai - admin_hours_with_ai
    
    # 4. IMPACTO MÉDICO (Horas de médicos parados à espera de quem não vem)
    # Faltas evitadas pela intervenção focada
    prevented_noshows = true_positives * intervention_success_rate
    
    # Horas médicas perdidas se não fizermos nada (todas as faltas acontecem)
    medical_hours_lost_no_ai = (total_actual_noshows * appointment_duration_mins) / 60
    
    # Horas médicas perdidas mesmo com a IA (Faltas não detetadas + Faltas não prevenidas)
    unprevented_noshows = false_negatives + (true_positives * (1 - intervention_success_rate))
    medical_hours_lost_with_ai = (unprevented_noshows * appointment_duration_mins) / 60
    
    medical_hours_saved = medical_hours_lost_no_ai - medical_hours_lost_with_ai
    
    print("==============================================================")
    print("IMPACTO OPERACIONAL: OTIMIZAÇÃO DE RECURSOS HOSPITALARES")
    print("==============================================================\n")
    print(f"Total de Pacientes Avaliados: {total_patients}")
    print(f"Ausências Reais: {total_actual_noshows}\n")
    
    print("> TEMPO ADMINISTRATIVO (Triagem e Confirmação de Consultas):")
    print(f"   - Sem IA (Ligar a todos):     {admin_hours_no_ai:.0f} horas de trabalho")
    print(f"   - Com IA (Apenas Alto Risco): {admin_hours_with_ai:.0f} horas de trabalho")
    print(f"   - Poupou-se à equipa:      {admin_hours_saved:.0f} horas!\n")
    
    print("> TEMPO MÉDICO (Horas de consultório vazias):")
    print(f"   - Sem IA (Todas as faltas):   {medical_hours_lost_no_ai:.0f} horas desperdiçadas")
    print(f"   - Com IA (Faltas reduzidas):  {medical_hours_lost_with_ai:.0f} horas desperdiçadas")
    print(f"   - Tempo médico recuperado: {medical_hours_saved:.0f} horas! (Aprox. {int(prevented_noshows)} consultas salvas)\n")
    print("================================================================")