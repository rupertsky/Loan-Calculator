import math
import argparse

parser = argparse.ArgumentParser(description="¡This is a Real Loan Calculator!")
parser.add_argument("--type", choices=["annuity", "diff"], help="¡You need to choose your loan type!")
parser.add_argument("--principal", type=int)
parser.add_argument("--periods", type=int)
parser.add_argument("--interest", type=float)
parser.add_argument("--payment", type=int)

args = parser.parse_args()
q = [args.type, args.principal, args.periods, args.interest, args.payment]

def lenght(arguments_list):
      long = 0
      for i in arguments_list:
            if i is not None:
                  long += 1
      return long

def negative(arguments_list):
      for i in arguments_list[1:]:
            if i is not None and i < 0:
                  print("Incorrect parameters")
                  break
      else:
            return True

m_counter = 0
m_sum = 0
if args.type == "diff" and args.payment is None and lenght(q) == 4 or lenght(q) == 5:
      if negative(q):
            m_i = args.interest / (12 * 100)
            while m_counter < args.periods:
                  m_counter += 1
                  month_p = math.ceil((args.principal / args.periods) + m_i * (args.principal - (args.principal * (m_counter -1) / args.periods)))
                  m_sum += month_p
                  over_pay = m_sum - args.principal
                  print(f"Month {m_counter}: payment is {month_p}")
            print(f"\nOverpayment = {over_pay}")
elif args.type == "annuity" and args.payment is None and lenght(q) == 4 or lenght(q) == 5:
      if negative(q):
            m_i = args.interest / (12 * 100)
            annuity = math.ceil(args.principal * (m_i * pow(1 + m_i, args.periods)) / (pow(1 + m_i, args.periods) - 1))
            over_pay = (annuity * args.periods) - args.principal
            print(f"Your annuity payment = {annuity}!")
            print(f"Overpayment = {over_pay}")
elif args.type == "annuity" and args.principal is None and lenght(q) == 4 or lenght(q) == 5:
      if negative(q):
            m_i = args.interest / (12 * 100)
            loan_p = math.floor(args.payment / ((m_i * pow(1 + m_i, args.periods)) / (pow(1 + m_i, args.periods) - 1)))
            over_pay = (args.payment * args.periods) - loan_p
            print(f"Your loan principal = {loan_p}!")
            print(f"Overpayment = {over_pay}")
elif args.type == "annuity" and args.periods is None and lenght(q) == 4 or lenght(q) == 5:
      if negative(q):
            m_i = args.interest / (12 * 100)
            months = math.log(args.payment / (args.payment - m_i * args.principal), 1 + m_i)
            months = math.ceil(months)
            years = months // 12
            m = months - (12 * years)
            over_pay = (months * args.payment) - args.principal
            if years != 0 and m != 0:
                  print(f"It will take {years} years and {m} months to repay this loan!")
            elif years != 0 and m == 0:
                  print(f"It will take {years} years to repay this loan!")
            elif years == 0 and m == 1:
                  print(f"It will take {m} month to repay this loan!")
            elif years == 1 and m == 0:
                  print(f"It will take {years} year to repay this loan!")
            else:
                  print(f"It will take {m} months to repay this loan!")
            print(f"Overpayment = {over_pay}")
else:
      print("Incorrect parameters")