from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional
import uuid


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
    """
    Абстрактный базовый класс для всех банковских счетов.
    Определяет общий интерфейс и свойства для всех типов счетов.
    """

    def __init__(
        self,
        account_id: str,
        owner: str,
        initial_balance: float = 0.0,
        status: AccountStatus = AccountStatus.ACTIVE
    ):
        """
        Инициализация абстрактного счёта.

        Args:
            account_id: Уникальный идентификатор счёта
            owner: Владелец счёта
            initial_balance: Начальный баланс (по умолчанию 0.0)
            status: Статус счёта (по умолчанию ACTIVE)
        """
        self._account_id = account_id
        self._owner = owner
        self._balance = initial_balance
        self._status = status

    @property
    def account_id(self) -> str:
        """Получить ID счёта."""
        return self._account_id

    @property
    def owner(self) -> str:
        """Получить владельца счёта."""
        return self._owner

    @property
    def balance(self) -> float:
        """Получить текущий баланс."""
        return self._balance

    @property
    def status(self) -> AccountStatus:
        """Получить статус счёта."""
        return self._status

    @abstractmethod
    def deposit(self, amount: float) -> None:
        """
        Внести средства на счёт.

        Args:
            amount: Сумма для внесения
        """
        pass

    @abstractmethod
    def withdraw(self, amount: float) -> None:
        """
        Снять средства со счёта.

        Args:
            amount: Сумма для снятия
        """
        pass

    @abstractmethod
    def get_account_info(self) -> dict:
        """
        Получить детальную информацию о счёте.

        Returns:
            Словарь с данными счёта
        """
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
        initial_balance: float = 0.0,
        account_id: Optional[str] = None,
        status: AccountStatus = AccountStatus.ACTIVE,
        currency: Currency = Currency.RUB
    ):
        """
        Инициализация банковского счёта.

        Args:
            owner: Владелец счёта
            initial_balance: Начальный баланс (по умолчанию 0.0)
            account_id: ID счёта (генерируется автоматически, если None)
            status: Статус счёта (по умолчанию ACTIVE)
            currency: Валюта счёта (по умолчанию RUB)

        Raises:
            InvalidOperationError: При ошибке валидации
        """
        # Валидация входных данных
        self._validate_owner(owner)
        self._validate_amount(initial_balance)

        # Генерация ID счёта, если не указан
        if account_id is None:
            account_id = self._generate_account_id()

        # Инициализация родительского класса
        super().__init__(account_id, owner, initial_balance, status)

        # Установка валюты
        self._currency = currency

    @staticmethod
    def _generate_account_id() -> str:
        """Генерация короткого UUID для ID счёта."""
        return str(uuid.uuid4())[:8].upper()

    @staticmethod
    def _validate_owner(owner: str) -> None:
        """Валидация имени владельца."""
        if not owner or not isinstance(owner, str) or not owner.strip():
            raise InvalidOperationError("Owner name must be a non-empty string")

    @staticmethod
    def _validate_amount(amount: float) -> None:
        """Валидация суммы транзакции."""
        if not isinstance(amount, (int, float)):
            raise InvalidOperationError("Amount must be a number")
        if amount < 0:
            raise InvalidOperationError("Amount cannot be negative")

    def _check_account_status(self) -> None:
        """Проверка статуса счёта перед операциями."""
        if self._status == AccountStatus.FROZEN:
            raise AccountFrozenError(self._account_id)
        if self._status == AccountStatus.CLOSED:
            raise AccountClosedError(self._account_id)

    def deposit(self, amount: float) -> None:
        """
        Внести средства на счёт.

        Args:
            amount: Сумма для внесения

        Raises:
            InvalidOperationError: Если сумма некорректна
            AccountFrozenError: Если счёт заморожен
            AccountClosedError: Если счёт закрыт
        """
        # Валидация суммы
        self._validate_amount(amount)
        # Проверка статуса счёта
        self._check_account_status()

        # Сумма должна быть больше нуля
        if amount == 0:
            raise InvalidOperationError("Deposit amount must be greater than zero")

        # Пополнение баланса
        self._balance += amount
        print(f"✓ Deposited {amount} {self._currency.value}. New balance: {self._balance} {self._currency.value}")

    def withdraw(self, amount: float) -> None:
        """
        Снять средства со счёта.

        Args:
            amount: Сумма для снятия

        Raises:
            InvalidOperationError: Если сумма некорректна
            InsufficientFundsError: Если недостаточно средств
            AccountFrozenError: Если счёт заморожен
            AccountClosedError: Если счёт закрыт
        """
        # Валидация суммы
        self._validate_amount(amount)
        # Проверка статуса счёта
        self._check_account_status()

        # Сумма должна быть больше нуля
        if amount == 0:
            raise InvalidOperationError("Withdrawal amount must be greater than zero")

        # Проверка достаточности средств
        if amount > self._balance:
            raise InsufficientFundsError(self._balance, amount)

        # Снятие средств
        self._balance -= amount
        print(f"✓ Withdrew {amount} {self._currency.value}. New balance: {self._balance} {self._currency.value}")

    def freeze_account(self) -> None:
        """Заморозить счёт."""
        # Нельзя заморозить закрытый счёт
        if self._status == AccountStatus.CLOSED:
            raise InvalidOperationError("Cannot freeze a closed account")
        self._status = AccountStatus.FROZEN
        print(f"Account {self._account_id} has been frozen.")

    def close_account(self) -> None:
        """Закрыть счёт."""
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
        """
        Получить детальную информацию о счёте.

        Returns:
            Словарь с полными данными счёта
        """
        return {
            "Account ID": self._account_id,
            "Owner": self._owner,
            "Balance": self._balance,
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