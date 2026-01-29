from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional, Dict, List, Tuple
import uuid
from decimal import Decimal, ROUND_HALF_UP, getcontext
import datetime

# Настройка Decimal для денег (из предыдущих дней)
getcontext().prec = 28
CENTS = Decimal("0.01")

# Предполагаем, что эти классы из Дня 1-2 уже определены
# Для полноты добавим минимальные заглушки, но используйте ваши версии
class AccountStatus(Enum):
    ACTIVE = "active"
    FROZEN = "frozen"
    CLOSED = "closed"

class AccountFrozenError(Exception):
    pass

class AccountClosedError(Exception):
    pass

class InvalidOperationError(Exception):
    pass

class InsufficientFundsError(Exception):
    pass

class AbstractAccount(ABC):
    @abstractmethod
    def deposit(self, amount: Decimal) -> None:
        pass

    @abstractmethod
    def withdraw(self, amount: Decimal) -> None:
        pass

    @abstractmethod
    def get_account_info(self) -> Dict:
        pass

# Заглушка для BankAccount (расширьте вашими Savings/Premium/Investment)
class BankAccount(AbstractAccount):
    def __init__(
        self,
        owner: str,
        initial_balance: str | float | int | Decimal = "0.00",
        account_id: Optional[str] = None,
    ):
        self._account_id = account_id or str(uuid.uuid4())[:8]
        self._owner = owner
        self._balance = self._to_money(initial_balance)
        self._status = AccountStatus.ACTIVE

    def _to_money(self, amount: str | float | int | Decimal) -> Decimal:
        if isinstance(amount, str):
            amount = Decimal(amount)
        else:
            amount = Decimal(str(amount))
        return amount.quantize(CENTS, rounding=ROUND_HALF_UP)

    def _check_account_status(self):
        if self._status == AccountStatus.CLOSED:
            raise AccountClosedError("Account is closed")
        if self._status == AccountStatus.FROZEN:
            raise AccountFrozenError("Account is frozen")

    def deposit(self, amount: Decimal) -> None:
        self._check_account_status()
        self._balance += self._to_money(amount)

    def withdraw(self, amount: Decimal) -> None:
        self._check_account_status()
        if self._balance < amount:
            raise InsufficientFundsError("Insufficient funds")
        self._balance -= self._to_money(amount)

    def freeze_account(self):
        self._status = AccountStatus.FROZEN

    def activate_account(self):
        self._status = AccountStatus.ACTIVE

    def close_account(self):
        if self._balance != 0:
            raise InvalidOperationError("Payout remaining balance first")
        self._status = AccountStatus.CLOSED

    def payout_remaining(self):
        amount = self._balance
        self._balance = Decimal("0.00")
        return amount

    def get_account_info(self) -> Dict:
        return {
            "type": self.__class__.__name__,
            "owner": self._owner,
            "id": self._account_id,
            "balance": float(self._balance),
            "status": self._status.value,
        }

    def __str__(self) -> str:
        last_four = self._account_id[-4:]
        return f"*{last_four} {self._owner} Bal: {self._balance} Status: {self._status.value}"

# Новые классы для Дня 3
class ClientStatus(Enum):
    ACTIVE = "active"
    BLOCKED = "blocked"
    SUSPICIOUS = "suspicious"

class Client:
    def __init__(
        self,
        full_name: str,
        id: Optional[str] = None,
        birth_date: Optional[str] = None,  # YYYY-MM-DD для проверки возраста
        contacts: Optional[Dict[str, str]] = None,
    ):
        self.id = id or str(uuid.uuid4())
        self.full_name = full_name
        self.contacts = contacts or {}
        self.accounts: List[str] = []  # список ID счетов
        self.status = ClientStatus.ACTIVE
        self.failed_attempts = 0
        self.is_blocked = False
        self.suspicious_actions = 0

        if birth_date:
            self._check_age(birth_date)

    def _check_age(self, birth_date: str):
        birth = datetime.datetime.strptime(birth_date, "%Y-%m-%d").date()
        age = datetime.date.today().year - birth.year
        if age < 18:
            raise ValueError("Client must be at least 18 years old")
        self.birth_date = birth_date

    def add_account(self, account_id: str):
        if account_id not in self.accounts:
            self.accounts.append(account_id)

    def authenticate(self, client_id: str, password: str) -> bool:  # упрощено, в реальности хэш
        if self.is_blocked:
            return False
        if client_id != self.id or password != "secret":  # заглушка пароля
            self.failed_attempts += 1
            if self.failed_attempts >= 3:
                self.is_blocked = True
                self.status = ClientStatus.BLOCKED
            return False
        self.failed_attempts = 0
        return True

    def mark_suspicious(self):
        self.suspicious_actions += 1
        if self.suspicious_actions >= 3:
            self.status = ClientStatus.SUSPICIOUS

class Bank:
    def __init__(self):
        self.clients: Dict[str, Client] = {}  # id -> Client
        self.accounts: Dict[str, BankAccount] = {}  # id -> Account
        self.suspicious_log = []  # для пометки

    def is_night_time(self) -> bool:
        now = datetime.datetime.now().hour
        return 0 <= now < 5  # 00:00 - 05:00

    def add_client(self, client: Client):
        if client.id in self.clients:
            raise ValueError("Client already exists")
        self.clients[client.id] = client

    def authenticate_client(self, client_id: str) -> Tuple[bool, Optional[Client]]:
        if client_id not in self.clients:
            return False, None
        client = self.clients[client_id]
        # authenticate требует password, но для банка упростим вызов
        if client.status == ClientStatus.ACTIVE and not client.is_blocked:
            return True, client
        return False, None

    def open_account(self, client_id: str, initial_balance: str = "0.00") -> str:
        if self.is_night_time():
            raise ValueError("Operations prohibited from 00:00 to 05:00")
        success, client = self.authenticate_client(client_id)
        if not success:
            client.mark_suspicious() if client else None
            raise ValueError("Client authentication failed")

        account = BankAccount(client.full_name, initial_balance)
        self.accounts[account._account_id] = account
        client.add_account(account._account_id)
        return account._account_id

    def close_account(self, account_id: str, client_id: str):
        if self.is_night_time():
            raise ValueError("Operations prohibited from 00:00 to 05:00")
        success, client = self.authenticate_client(client_id)
        if not success:
            raise ValueError("Client authentication failed")

        if account_id not in client.accounts:
            raise ValueError("Account not owned by client")
        if account_id not in self.accounts:
            raise ValueError("Account not found")

        account = self.accounts[account_id]
        account.close_account()
        del self.accounts[account_id]
        client.accounts.remove(account_id)

    def freeze_account(self, account_id: str, client_id: str):
        success, client = self.authenticate_client(client_id)
        if not success:
            raise ValueError("Client authentication failed")
        if account_id not in client.accounts:
            raise ValueError("Not your account")
        self.accounts[account_id].freeze_account()

    def unfreeze_account(self, account_id: str, client_id: str):
        success, client = self.authenticate_client(client_id)
        if not success:
            raise ValueError("Client authentication failed")
        self.accounts[account_id].activate_account()

    def search_accounts(self, client_id: str) -> List[Dict]:
        success, client = self.authenticate_client(client_id)
        if not success:
            return []
        return [self.accounts[acc_id].get_account_info() for acc_id in client.accounts]

    def get_total_balance(self, client_id: str) -> Decimal:
        success, client = self.authenticate_client(client_id)
        if not success:
            raise ValueError("Authentication failed")
        total = Decimal("0.00")
        for acc_id in client.accounts:
            if acc_id in self.accounts:
                total += self.accounts[acc_id]._balance
        return total

    def get_clients_ranking(self) -> List[Tuple[str, Decimal]]:
        ranking = []
        for client in self.clients.values():
            if client.status == ClientStatus.ACTIVE:
                total = sum(self.accounts[acc_id]._balance for acc_id in client.accounts if acc_id in self.accounts)
                ranking.append((client.full_name, total))
        return sorted(ranking, key=lambda x: x[1], reverse=True)

# Тестирование
if __name__ == "__main__":
    bank = Bank()

    # Создание клиентов
    client1 = Client("Иван Иванов", birth_date="1990-01-01", contacts={"phone": "+7-999-001"})
    client2 = Client("Анна Петрова", birth_date="2000-05-15", contacts={"email": "anna@example.com"})

    bank.add_client(client1)
    bank.add_client(client2)

    # Открытие счетов
    acc1 = bank.open_account(client1.id, "10000.00")
    acc2 = bank.open_account(client1.id, "5000.00")
    acc3 = bank.open_account(client2.id, "20000.00")

    print("Счета Ивана:", bank.search_accounts(client1.id))
    print("Общий баланс Ивана:", bank.get_total_balance(client1.id))

    # Заморозка
    bank.freeze_account(acc1, client1.id)
    try:
        bank.accounts[acc1].deposit(100)
    except AccountFrozenError:
        print("Счет заморожен - OK")

    bank.unfreeze_account(acc1, client1.id)

    # Неудачные попытки аутентификации
    print(bank.authenticate_client("wrong_id"))  # False
    # В реальности несколько вызовов для 3 попыток

    # Рейтинг
    print("Рейтинг клиентов:", bank.get_clients_ranking())

    print("\n=== День 3 готов! ===")