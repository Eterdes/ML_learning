from enum import Enum
from datetime import datetime
import logging

class Transaction:
    def __init__(self, id: str, amount: float, from_client: str, to_account: str, 
                 is_night: bool = False, to_new: bool = False):
        self.id = id
        self.amount = amount
        self.from_client = from_client
        self.to_account = to_account
        self.is_night = is_night
        self.to_new = to_new
        self.blocked = False


class AuditLevel(Enum):
    INFO = "INFO"
    WARNING = "WARNING"


class AuditLog:
    def __init__(self):
        self.logs = []
        logging.basicConfig(filename='audit.log', level=logging.INFO)
        self.logger = logging.getLogger('bank')

    def log(self, level: AuditLevel, message: str):
        # 1. Создаём словарь с логом
        entry = {
            'time': datetime.now().isoformat(),
            'level': level.value,
            'message': message
        }
        # 2. Сохраняем в память
        self.logs.append(entry)
        # 3. Пишем в файл
        self.logger.info(f"{level.value}: {message}")
        # 4. Показываем на экран
        print(f"[{level.value}] {message}")


    def filter(self, min_level: AuditLevel = AuditLevel.INFO):
        level_order = {
            AuditLevel.INFO: 1,
            AuditLevel.WARNING: 2
        }
        result = []
        for log in self.logs:
            if level_order[AuditLevel(log['level'])] >= level_order[min_level]:
                result.append(log)
        return result
    
class RiskLevel(Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM" 
    HIGH = "HIGH"

class RiskAnalyzer:
    def __init__(self):
        self.audit = AuditLog()  
    
    def analyze(self, amount: float, is_night: bool, to_new_account: bool = False):
        risks = 0
        
        if amount > 100000:
            risks += 1
            self.audit.log(AuditLevel.WARNING, f"🚨 Большая сумма: {amount}")
        
        if is_night:
            risks += 1
            self.audit.log(AuditLevel.WARNING, "🌙 Операция ночью!")
        
        if to_new_account:
            risks += 1
            self.audit.log(AuditLevel.WARNING, "🆕 Новый счёт!")
        
        if risks == 0:
            level = RiskLevel.LOW
        elif risks == 1:
            level = RiskLevel.MEDIUM
        else:
            level = RiskLevel.HIGH
            
        self.audit.log(AuditLevel.INFO, f"Итог: {level.value} ({risks} рисков)")
        return level
    
    def should_block(self, risk: RiskLevel) -> bool:
        if risk == RiskLevel.HIGH:
            self.audit.log(AuditLevel.WARNING, "🚫 БЛОКИРУЕМ операцию!")
            return True
        return False

class AuditReporter:
    def __init__(self, analyzer: RiskAnalyzer):
        self.analyzer = analyzer
    
    def report(self):
        """Отчёт: подозрительные + статистика"""
        all_logs = self.analyzer.audit.logs
        warnings = self.analyzer.audit.filter(AuditLevel.WARNING)
        
        suspicious = [log for log in warnings if "БЛОКИРУЕМ" in log['message']]
        
        print("\n📊 ОТЧЁТ АУДИТА")
        print(f"Всего логов: {len(all_logs)}")
        print(f"Предупреждений: {len(warnings)}")
        print(f"Заблокировано: {len(suspicious)}")
        
        if suspicious:
            print("🚨 Подозрительные:")
            for s in suspicious:
                print(f"  {s['message']}")

class Client:
    def __init__(self, id: str, name: str):
        self.id = id
        self.name = name
        self.accounts = []  # список счетов
        
class SimpleBank:
    def __init__(self):
        self.clients = {}  # id → Client
        self.accounts = {}  # номер → баланс
        self.risk_analyzer = RiskAnalyzer()
        self.transactions = []
    
    def add_client(self, client: Client):
        self.clients[client.id] = client
        print(f"✅ Клиент {client.name} добавлен")
    
    def open_account(self, client_id: str, account_num: str, balance: float = 0):
        self.accounts[account_num] = balance
        self.clients[client_id].accounts.append(account_num)
        print(f"✅ Счёт {account_num} открыт для {self.clients[client_id].name}")
    
    def process_transaction(self, tx: Transaction):
        risk = self.risk_analyzer.analyze(
            tx.amount, 
            tx.is_night, 
            tx.to_new
        )
        
        if self.risk_analyzer.should_block(risk):
            tx.blocked = True
            self.risk_analyzer.audit.log(AuditLevel.WARNING, f"🚫 Tx {tx.id} заблокирована")
        else:
            print(f"✅ Tx {tx.id} выполнена: {tx.amount}")
        
        self.transactions.append(tx)

    def show_client(self, client_id: str):
        """Показать счета клиента"""
        client = self.clients.get(client_id)
        if not client:
            print("❌ Клиент не найден")
            return
        
        print(f"\n👤 {client.name} (ID: {client.id})")
        print(f"Счета: {', '.join(client.accounts)}")
    
    def client_history(self, client_id: str):
        """История tx клиента"""
        client_tx = [tx for tx in self.transactions if tx.from_client == self.clients[client_id].name]
        blocked = [tx for tx in client_tx if tx.blocked]
        
        print(f"\n📋 {self.clients[client_id].name}: {len(client_tx)} tx")
        print(f"Заблокировано: {len(blocked)}")
        if blocked:
            print("🚨 Заблокированные:")
            for tx in blocked:
                print(f"  {tx.id}: {tx.amount}")

    def final_report(self):
        """Топ-3 клиентов + статистика"""
        print("\n🏆 ФИНАЛЬНЫЙ ОТЧЁТ БАНКА")
        
        # Статистика tx
        total_tx = len(self.transactions)
        blocked_tx = len([tx for tx in self.transactions if tx.blocked])
        print(f"📈 Всего tx: {total_tx}")
        print(f"🚫 Заблокировано: {blocked_tx} ({blocked_tx/total_tx*100:.1f}%)")
        
        # Топ-3 по tx
        client_tx_count = {}
        for tx in self.transactions:
            client_tx_count[tx.from_client] = client_tx_count.get(tx.from_client, 0) + 1
        
        top_clients = sorted(client_tx_count.items(), key=lambda x: x[1], reverse=True)[:3]
        print("🥇 Топ-3 активных клиентов:")
        for i, (name, count) in enumerate(top_clients, 1):
            print(f"  {i}. {name}: {count} tx")
        
        # Общий баланс (сумма всех счетов)
        total_balance = sum(self.accounts.values())
        print(f"💰 Общий баланс банка: {total_balance:,.0f}")
        
        # Финальный аудит
        reporter = AuditReporter(self.risk_analyzer)
        reporter.report()


def simulate_transactions(bank: SimpleBank, count: int = 50):
    """Создаёт 50 tx: 80% норм, 15% подозр, 5% HIGH"""
    import random
    
    clients = list(bank.clients.keys())
    accounts = list(bank.accounts.keys())
    
    for i in range(count):
        client_from = random.choice(clients)
        account_to = random.choice(accounts)
        amount = random.uniform(100, 50000)  # обычно
        
        # 20% подозрительных
        is_suspicious = random.random() < 0.2
        if is_suspicious:
            amount = random.uniform(100000, 500000)  # большая!
        
        is_night = random.random() < 0.1  # 10% ночью
        is_new = account_to not in ["acc1", "acc2"]  # новые счета
        
        tx = Transaction(
            id=f"tx{i+1}",
            amount=amount,
            from_client=bank.clients[client_from].name,
            to_account=account_to,
            is_night=is_night,
            to_new=is_new
        )
        
        bank.process_transaction(tx)
    
    print(f"\n🏦 Симуляция: {count} транзакций завершена!")





# ДЕМО БАНКА
bank = SimpleBank()

# ИНИЦИАЛИЗАЦИЯ (ТЗ: 5-10 клиентов, 10-15 счетов)
print("🏦 ИНИЦИАЛИЗАЦИЯ БАНКА")

clients_data = [
    ("1", "Иванов Иван"), ("2", "Мария Петрова"), ("3", "Петров Сергей"),
    ("4", "Сидорова Анна"), ("5", "Козлов Дмитрий"), ("6", "Смирнова Ольга"),
    ("7", "Васильев Алексей"), ("8", "Новикова Екатерина"), ("9", "Морозов Олег"),
    ("10", "Федорова Елена")
]

for client_id, name in clients_data:
    client = Client(client_id, name)
    bank.add_client(client)

# 12 счетов
account_data = [
    ("1", "acc001", 100000), ("1", "acc002", 50000),
    ("2", "acc003", 75000), 
    ("3", "acc004", 20000), ("3", "acc005", 30000),
    ("4", "acc006", 150000),
    ("5", "acc007", 80000),
    ("6", "acc008", 120000), ("6", "acc009", 40000),
    ("7", "acc010", 95000),
    ("8", "acc011", 60000),
    ("9", "acc012", 110000)
]

for client_id, acc_num, balance in account_data:
    bank.open_account(client_id, acc_num, balance)


simulate_transactions(bank, 20)  # 20 tx для начала

print("\n=== ПОЛЬЗОВАТЕЛЬСКИЕ КОМАНДЫ ===")
bank.show_client("1")           # счета Иванова
bank.client_history("1")        # история Иванова
bank.client_history("2")        # история Марии


reporter = AuditReporter(bank.risk_analyzer)
reporter.report()

bank.final_report()