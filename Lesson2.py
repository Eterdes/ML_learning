from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional, Dict
import uuid
from decimal import Decimal, ROUND_HALF_UP, getcontext

# Настройка Decimal для денег
getcontext().prec = 28
CENTS = Decimal("0.01")

# Пользовательские классы исключений
class AccountFrozenError(Exception):
    """Исключение при попытке операций с замороженным счётом."""
    def __init__(self, account_id: str):
        self.account_id = account_id
        super().__init__(f"Account {account_id} is frozen. Operations are not allowed.")

class AccountClosedError(Exception):
    """Исключение при попытке операций с закрытым счётом."""
    def __init__(self, account_id: str):
        self.account_id = account_id
        super().__init__(f"Account {account_id} is closed. Operations are not allowed.")

class InvalidOperationError(Exception):
    """Исключение для недопустимых операций."""
    def __init__(self, message: str):
        super().__init__(f"Invalid operation: {message}")

class InsufficientFundsError(Exception):
    """Исключение при недостаточном балансе."""
    def __init__(self, balance: float, amount: float):
        self.balance = balance
        self.amount = amount
        super().__init__(
            f"Insufficient funds. Balance: {balance}, Requested: {amount}"
        )

# Перечисления
class AccountStatus(Enum):
    """Статусы банковского счёта."""
    ACTIVE = "active"
    FROZEN = "frozen"
    CLOSED = "closed"

class Currency(Enum):
    """Поддерживаемые валюты."""
    RUB = "RUB"
    USD = "USD"
    EUR = "EUR"
    KZT = "KZT"
    CNY = "CNY"

class PortfolioAsset(Enum):
    """Типы активов в портфеле."""
    STOCKS = "stocks"
    BONDS = "bonds"
    ETF = "etf"

# Абстрактный базовый класс
class AbstractAccount(ABC):
    def __init__(
        self,
        account_id: str,
        owner: str,
        initial_balance: Decimal = Decimal("0.00"),
        status: AccountStatus = AccountStatus.ACTIVE
    ):
        self._account_id = account_id
        self._owner = owner
        self._balance: Decimal = initial_balance
        self._status = status

    @property
    def account_id(self) -> str:
        return self._account_id

    @property
    def owner(self) -> str:
        return self._owner

    @property
    def balance(self) -> Decimal:
        return self._balance

    @property
    def status(self) -> AccountStatus:
        return self._status

    @abstractmethod
    def deposit(self, amount: Decimal) -> None:
        pass

    @abstractmethod
    def withdraw(self, amount: Decimal) -> None:
        pass

    @abstractmethod
    def get_account_info(self) -> dict:
        pass

# Базовый класс BankAccount
class BankAccount(AbstractAccount):
    def __init__(
        self,
        owner: str,
        initial_balance: float | Decimal | int | str = "0.00",
        account_id: Optional[str] = None,
        status: AccountStatus = AccountStatus.ACTIVE,
        currency: Currency = Currency.RUB
    ):
        self._validate_owner(owner)
        initial_balance_dec = self._to_money(initial_balance)
        self._validate_amount(initial_balance_dec)

        if account_id is None:
            account_id = self._generate_account_id()

        self._money_zero = Decimal("0.00")
        super().__init__(account_id, owner, initial_balance_dec, status)
        self._currency = currency

    @staticmethod
    def _generate_account_id() -> str:
        return str(uuid.uuid4())[:8].upper()

    @staticmethod
    def _validate_owner(owner: str) -> None:
        if not owner or not isinstance(owner, str) or not owner.strip():
            raise InvalidOperationError("Owner name must be a non-empty string")

    @staticmethod
    def _to_money(amount: float | Decimal | int | str) -> Decimal:
        if isinstance(amount, Decimal):
            dec = amount
        else:
            dec = Decimal(str(amount))
        return dec.quantize(CENTS, rounding=ROUND_HALF_UP)

    @staticmethod
    def _validate_amount(amount: Decimal) -> None:
        if not isinstance(amount, Decimal):
            raise InvalidOperationError("Amount must be a Decimal")
        if amount < Decimal("0.00"):
            raise InvalidOperationError("Amount cannot be negative")

    def _check_account_status(self) -> None:
        if self._status == AccountStatus.FROZEN:
            raise AccountFrozenError(self._account_id)
        if self._status == AccountStatus.CLOSED:
            raise AccountClosedError(self._account_id)

    def deposit(self, amount: float | Decimal | int | str) -> None:
        amount_dec = self._to_money(amount)
        self._validate_amount(amount_dec)
        self._check_account_status()

        if amount_dec == self._money_zero:
            raise InvalidOperationError("Deposit amount must be greater than zero")

        self._balance = (self._balance + amount_dec).quantize(CENTS, rounding=ROUND_HALF_UP)
        print(f"✓ Deposited {amount_dec} {self._currency.value}. New balance: {self._balance} {self._currency.value}")

    def withdraw(self, amount: float | Decimal | int | str) -> None:
        amount_dec = self._to_money(amount)
        self._validate_amount(amount_dec)
        self._check_account_status()

        if amount_dec == self._money_zero:
            raise InvalidOperationError("Withdrawal amount must be greater than zero")

        if amount_dec > self._balance:
            raise InsufficientFundsError(float(self._balance), float(amount_dec))

        self._balance = (self._balance - amount_dec).quantize(CENTS, rounding=ROUND_HALF_UP)
        print(f"✓ Withdrew {amount_dec} {self._currency.value}. New balance: {self._balance} {self._currency.value}")

    def freeze_account(self) -> None:
        if self._status == AccountStatus.CLOSED:
            raise InvalidOperationError("Cannot freeze a closed account")
        self._status = AccountStatus.FROZEN
        print(f"Account {self._account_id} has been frozen.")

    def payout_remaining(self) -> Decimal:
        self._check_account_status()
        if self._balance == self._money_zero:
            return self._money_zero
        amount = self._balance
        self._balance = self._money_zero
        print(f"✓ Paid out {amount} {self._currency.value}. New balance: {self._balance} {self._currency.value}")
        return amount

    def close_account(self) -> None:
        if self._balance != self._money_zero:
            raise InvalidOperationError(
                "Cannot close account with non-zero balance. Use payout_remaining() first."
            )
        self._status = AccountStatus.CLOSED
        print(f"Account {self._account_id} has been closed.")

    def activate_account(self) -> None:
        if self._status == AccountStatus.CLOSED:
            raise InvalidOperationError("Cannot activate a closed account")
        self._status = AccountStatus.ACTIVE
        print(f"Account {self._account_id} has been activated.")

    def get_account_info(self) -> dict:
        return {
            "Account ID": self._account_id,
            "Owner": self._owner,
            "Balance": float(self._balance),
            "Currency": self._currency.value,
            "Status": self._status.value,
            "Account Type": self.__class__.__name__
        }

    def __str__(self) -> str:
        last_four = self._account_id[-4:]
        return (
            f"Type: {self.__class__.__name__} | "
            f"Client: {self._owner} | "
            f"Number: ****{last_four} | "
            f"Status: {self._status.value} | "
            f"Balance: {self._balance} {self._currency.value}"
        )

# День 2: Продвинутые типы счетов
class SavingsAccount(BankAccount):
    def __init__(
        self,
        owner: str,
        initial_balance: float | Decimal | int | str = "0.00",
        account_id: Optional[str] = None,
        min_balance: Decimal = Decimal("1000.00"),
        monthly_interest_rate: Decimal = Decimal("0.005")
    ):
        super().__init__(owner, initial_balance, account_id)
        self._min_balance = self._to_money(min_balance)
        self._monthly_interest_rate = self._to_money(monthly_interest_rate)

    def withdraw(self, amount: float | Decimal | int | str) -> None:
        amount_dec = self._to_money(amount)
        projected_balance = self._balance - amount_dec
        if projected_balance < self._min_balance:
            raise InvalidOperationError(
                f"Withdrawal denied. Minimum balance {self._min_balance} required."
            )
        super().withdraw(amount_dec)

    def apply_monthly_interest(self) -> Decimal:
        eligible_balance = max(self._balance - self._min_balance, self._money_zero)
        interest = eligible_balance * self._monthly_interest_rate
        self._balance += interest
        print(f"✓ Interest applied: {interest} {self._currency.value}")
        return interest

    def get_account_info(self) -> dict:
        info = super().get_account_info()
        info.update({
            "Account Type": "SavingsAccount",
            "Min Balance": float(self._min_balance),
            "Monthly Rate": float(self._monthly_interest_rate)
        })
        return info

    def __str__(self) -> str:
        last_four = self._account_id[-4:]
        return (
            f"Savings ****{last_four} | {self._owner} | "
            f"Bal: {self._balance} | Min: {self._min_balance} | "
            f"Status: {self._status.value}"
        )

class PremiumAccount(BankAccount):
    def __init__(
        self,
        owner: str,
        initial_balance: float | Decimal | int | str = "0.00",
        account_id: Optional[str] = None,
        overdraft_limit: Decimal = Decimal("-5000.00"),
        monthly_fee: Decimal = Decimal("100.00")
    ):
        super().__init__(owner, initial_balance, account_id)
        self._overdraft_limit = self._to_money(overdraft_limit)
        self._monthly_fee = self._to_money(monthly_fee)

    def withdraw(self, amount: float | Decimal | int | str) -> None:
        amount_dec = self._to_money(amount)
        projected_balance = self._balance - amount_dec
        if projected_balance < self._overdraft_limit:
            raise InsufficientFundsError(float(self._balance), float(amount_dec))
        super().withdraw(amount_dec)
        if self._balance < self._money_zero:
            print(f"⚠️ Overdraft used. Available limit: {self._overdraft_limit}")

    def apply_monthly_fee(self) -> None:
        if self._balance < self._monthly_fee:
            raise InsufficientFundsError(float(self._balance), float(self._monthly_fee))
        self._balance -= self._monthly_fee
        print(f"💳 Monthly fee applied: {self._monthly_fee}")

    def get_account_info(self) -> dict:
        info = super().get_account_info()
        info.update({
            "Account Type": "PremiumAccount",
            "Overdraft Limit": float(self._overdraft_limit),
            "Monthly Fee": float(self._monthly_fee)
        })
        return info

    def __str__(self) -> str:
        last_four = self._account_id[-4:]
        overdraft_used = min(self._balance, self._money_zero)
        return (
            f"Premium ****{last_four} | {self._owner} | "
            f"Bal: {self._balance} | OD: {overdraft_used} | "
            f"Status: {self._status.value}"
        )

class InvestmentAccount(BankAccount):
    def __init__(
        self,
        owner: str,
        initial_balance: float | Decimal | int | str = "0.00",
        account_id: Optional[str] = None,
        portfolio: Dict[PortfolioAsset, Decimal] = None
    ):
        super().__init__(owner, initial_balance, account_id)
        self._portfolio = portfolio or {
            PortfolioAsset.STOCKS: Decimal("0.40"),
            PortfolioAsset.BONDS: Decimal("0.40"),
            PortfolioAsset.ETF: Decimal("0.20")
        }

    def project_yearly_growth(self, expected_returns: Dict[PortfolioAsset, Decimal]) -> Decimal:
        growth = self._money_zero
        for asset, weight in self._portfolio.items():
            growth += weight * expected_returns.get(asset, Decimal("0.05"))
        projected_balance = self._balance * (Decimal("1") + growth)
        print(f"📈 Projected yearly growth: +{growth:.1%} → {projected_balance}")
        return projected_balance

    def get_account_info(self) -> dict:
        info = super().get_account_info()
        info.update({
            "Account Type": "InvestmentAccount",
            "Portfolio": {k.value: float(v) for k, v in self._portfolio.items()}
        })
        return info

    def __str__(self) -> str:
        last_four = self._account_id[-4:]
        port_str = "/".join(f"{k.value[0]:.0%}" for k in self._portfolio.keys())
        return (
            f"Invest ****{last_four} | {self._owner} | "
            f"Bal: {self._balance} | Portf: {port_str} | "
            f"Status: {self._status.value}"
        )

# Тестирование всех типов
if __name__ == "__main__":
    print("=== День 2: Все типы счетов ===\n")
    
    # Savings
    savings = SavingsAccount("Анна", 5000, min_balance=2000)
    print(savings)
    savings.apply_monthly_interest()
    
    # Premium
    premium = PremiumAccount("Борис", 3000, overdraft_limit=-2000)
    print(premium)
    premium.withdraw(4500)
    
    # Investment
    invest = InvestmentAccount("Вера", 10000)
    print(invest)
    returns = {PortfolioAsset.STOCKS: Decimal('0.12'), 
               PortfolioAsset.BONDS: Decimal('0.04'), 
               PortfolioAsset.ETF: Decimal('0.08')}
    invest.project_yearly_growth(returns)
    
    print("\n=== Готово! ===")
