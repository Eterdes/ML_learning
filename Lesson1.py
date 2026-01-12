from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional
import uuid
from decimal import Decimal, ROUND_HALF_UP, getcontext

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


# Конкретная реализация
class BankAccount(AbstractAccount):
    """
    Конкретная реализация банковского счёта с расширенными возможностями.
    Включает валидацию, проверку статусов и поддержку нескольких валют.
    """
    def __init__(
        self,
        owner: str,
        initial_balance: float | Decimal | int | str = "0.00",
        account_id: Optional[str] = None,
        status: AccountStatus = AccountStatus.ACTIVE,
        currency: Currency = Currency.RUB
    ):
        self._validate_owner(owner)
        
        # Приводим к Decimal через строку
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
        return dec.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

    @staticmethod
    def _validate_amount(amount: Decimal) -> None:
        if not isinstance(amount, Decimal):
            raise InvalidOperationError("Amount must be a Decimal")
        if amount < Decimal("0.00"):
            raise InvalidOperationError("Amount cannot be negative")


    def _check_account_status(self) -> None:
        """Проверка статуса счёта перед операциями."""
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

        self._balance = (self._balance + amount_dec).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        print(f"✓ Deposited {amount_dec} {self._currency.value}. New balance: {self._balance} {self._currency.value}")


    def withdraw(self, amount: float | Decimal | int | str) -> None:
        amount_dec = self._to_money(amount)
        self._validate_amount(amount_dec)
        self._check_account_status()

        if amount_dec == self._money_zero:
            raise InvalidOperationError("Withdrawal amount must be greater than zero")

        if amount_dec > self._balance:
            raise InsufficientFundsError(float(self._balance), float(amount_dec))

        self._balance = (self._balance - amount_dec).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        print(f"✓ Withdrew {amount_dec} {self._currency.value}. New balance: {self._balance} {self._currency.value}")


    def freeze_account(self) -> None:
        """Заморозить счёт."""
        # Нельзя заморозить закрытый счёт
        if self._status == AccountStatus.CLOSED:
            raise InvalidOperationError("Cannot freeze a closed account")
        self._status = AccountStatus.FROZEN
        print(f"Account {self._account_id} has been frozen.")

    def payout_remaining(self) -> Decimal:
        """Вывести весь остаток (для закрытия счёта)."""
        self._check_account_status()
        if self._balance == self._money_zero:
            return self._money_zero
        amount = self._balance
        self._balance = self._money_zero
        print(f"✓ Paid out {amount} {self._currency.value}. New balance: {self._balance} {self._currency.value}")
        return amount

    def close_account(self) -> None:
        """Закрыть счёт (только при нулевом балансе)."""
        if self._balance != self._money_zero:
            raise InvalidOperationError(
                "Cannot close account with non-zero balance. Use payout_remaining() first."
            )
        self._status = AccountStatus.CLOSED
        print(f"Account {self._account_id} has been closed.")

    def activate_account(self) -> None:
        """Активировать счёт."""
        # Нельзя активировать закрытый счёт
        if self._status == AccountStatus.CLOSED:
            raise InvalidOperationError("Cannot activate a closed account")
        self._status = AccountStatus.ACTIVE
        print(f"Account {self._account_id} has been activated.")

    def get_account_info(self) -> dict:
        return {
            "Account ID": self._account_id,
            "Owner": self._owner,
            "Balance": float(self._balance),  # для JSON-сериализации
            "Currency": self._currency.value,
            "Status": self._status.value,
            "Account Type": self.__class__.__name__
        }

    def __str__(self) -> str:
        """
        Строковое представление счёта.

        Returns:
            Отформатированная строка с информацией о счёте
        """
        # Получаем последние 4 цифры номера счёта
        last_four = self._account_id[-4:]

        return (
            f"Type: {self.__class__.__name__} | "
            f"Client: {self._owner} | "
            f"Number: ****{last_four} | "
            f"Status: {self._status.value} | "
            f"Balance: {self._balance} {self._currency.value}"
        )