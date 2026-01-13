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

# Заглушка для BankAccount 
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
        
    @property
    def balance(self):
        return self._balance

    @property 
    def status(self):
        return self._status.value



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
    
class PremiumAccount(BankAccount):
    """Премиум счёт — может уходить в минус"""
    pass  # наследует все методы от BankAccount


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
    
    def find_account(self, account_id: str) -> Optional[BankAccount]:
        """Найти счёт по ID"""
        return self.accounts.get(account_id)

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


# День 4: Система транзакций

# 1. Enum для транзакций
class TransactionType(Enum):
    TRANSFER = "transfer"
    DEPOSIT = "deposit" 
    WITHDRAW = "withdraw"

class TransactionStatus(Enum):
    NEW = "new"
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"

# 2. Класс Transaction
class Transaction:
    def __init__(
        self,
        tx_type: TransactionType,
        amount: Decimal,
        currency: str = "RUB",
        from_account_id: Optional[str] = None,
        to_account_id: Optional[str] = None,
        fee: Decimal = Decimal("0.00")
    ):
        self.transaction_id = str(uuid.uuid4())
        self.tx_type = tx_type
        self.amount = amount
        self.currency = currency
        self.fee = fee
        self.from_account_id = from_account_id
        self.to_account_id = to_account_id
        self.status = TransactionStatus.NEW
        self.failure_reason: Optional[str] = None
        self.created_at = datetime.datetime.now()
        self.processed_at = None
    
    def __repr__(self):
        return f"Transaction({self.tx_type.value}, {self.amount}, {self.status.value})"

# 3. Очередь транзакций
class TransactionQueue:
    def __init__(self):
        self.queue = []
    
    def add(self, transaction, priority=0, run_at=None):
        if run_at is None:
            run_at = datetime.datetime.now()
        item = (-priority, run_at, transaction)
        self.queue.append(item)
    
    def get_next(self):
        best_index = -1
        best_priority = float('inf')
        now = datetime.datetime.now()
        
        for i, item in enumerate(self.queue):
            if item[1] <= now and item[0] < best_priority:
                best_priority = item[0]
                best_index = i
        
        if best_index == -1:
            return None
        
        return self.queue.pop(best_index)
    
    def cancel(self, transaction_id):
        self.queue = [item for item in self.queue 
                     if item[2].transaction_id != transaction_id]
    
    def size(self):
        return len(self.queue)

# 4. Процессор транзакций
class TransactionProcessor:
    def __init__(self, bank):
        self.bank = bank
    
    def process_transaction(self, tx: Transaction):
        if tx.status != TransactionStatus.NEW:
            return
        
        try:
            # Найти счета
            from_account = self.bank.find_account(tx.from_account_id)
            to_account = self.bank.find_account(tx.to_account_id) if tx.to_account_id else None
            
            # Проверка заморозки
            if from_account.status == "frozen":
                raise Exception("Счёт отправителя заморожен")
            
            total_amount = tx.amount + tx.fee
            
            # Правила баланса
            if tx.tx_type in [TransactionType.TRANSFER, TransactionType.WITHDRAW]:
                if from_account.balance < total_amount:
                    # Премиум может уходить в минус
                    if not isinstance(from_account, PremiumAccount):
                        raise Exception("Недостаточно средств")
            
            # Выполнить операцию
            if tx.tx_type == TransactionType.DEPOSIT and to_account:
                to_account.deposit(tx.amount)
            elif tx.tx_type == TransactionType.WITHDRAW:
                from_account.withdraw(total_amount)
            elif tx.tx_type == TransactionType.TRANSFER and to_account:
                from_account.withdraw(total_amount)
                to_account.deposit(tx.amount)
            
            # Успех
            tx.status = TransactionStatus.COMPLETED
            tx.processed_at = datetime.datetime.now()
            
        except Exception as e:
            tx.status = TransactionStatus.FAILED
            tx.failure_reason = str(e)

# 5. ТЕСТИРОВАНИЕ
if __name__ == "__main__":
    bank = Bank()
    
    # Создаём счета напрямую
    acc1 = PremiumAccount("Клиент1", Decimal("1500.00"))
    acc2 = BankAccount("Клиент2", Decimal("800.00"))
    bank.accounts[acc1._account_id] = acc1
    bank.accounts[acc2._account_id] = acc2
    
    print(f"Начало: acc1={acc1._balance}, acc2={acc2._balance}")
    
    # 10 транзакций ИСПРАВЛЕННЫХ
    transactions = [
        Transaction(TransactionType.DEPOSIT, Decimal("100"), to_account_id=acc1._account_id),
        Transaction(TransactionType.WITHDRAW, Decimal("50"), from_account_id=acc1._account_id),
        Transaction(TransactionType.TRANSFER, Decimal("200"), from_account_id=acc1._account_id, to_account_id=acc2._account_id),
        Transaction(TransactionType.WITHDRAW, Decimal("300"), from_account_id=acc1._account_id),
        Transaction(TransactionType.DEPOSIT, Decimal("150"), to_account_id=acc2._account_id),
        Transaction(TransactionType.TRANSFER, Decimal("100"), from_account_id=acc2._account_id, to_account_id=acc1._account_id),
        Transaction(TransactionType.WITHDRAW, Decimal("1000"), from_account_id=acc2._account_id),  # FAIL
        Transaction(TransactionType.DEPOSIT, Decimal("250"), to_account_id=acc1._account_id, priority=10),
        Transaction(TransactionType.WITHDRAW, Decimal("75"), from_account_id=acc1._account_id),
        Transaction(TransactionType.TRANSFER, Decimal("50"), from_account_id=acc1._account_id, to_account_id=acc2._account_id),
    ]
    
    queue = TransactionQueue()
    for tx in transactions:
        queue.add(tx, priority=len(transactions) - transactions.index(tx))
    
    print(f"🚀 {queue.size()} транзакций в очереди")
    
    processor = TransactionProcessor(bank)
    while queue.size() > 0:
        tx = queue.get_next()
        processor.process_transaction(tx)
        result = f"{tx.status.value}: {tx.tx_type.value} {tx.amount}"
        if tx.failure_reason:
            result += f" ❌ {tx.failure_reason}"
        print(result)
    
    print(f"\n✅ КОНЕЦ: acc1={acc1._balance}, acc2={acc2._balance}")