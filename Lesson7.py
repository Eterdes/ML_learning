# Lesson7.py — День 7 (БЕЗ pandas - только стандартные библиотеки + matplotlib)
from enum import Enum
from datetime import datetime
from datetime import timedelta
import logging
import json
import matplotlib.pyplot as plt
import os
import numpy as np
import csv  # стандартная для CSV
from collections import Counter
import random

# ТВОИ КЛАССЫ (полностью сохранены - без изменений)
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
        entry = {
            'time': datetime.now().isoformat(),
            'level': level.value,
            'message': message
        }
        self.logs.append(entry)
        self.logger.info(f"{level.value}: {message}")
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
        all_logs = self.analyzer.audit.logs
        warnings = self.analyzer.audit.filter(AuditLevel.WARNING)
        suspicious = [log for log in warnings if "БЛОКИРУЕМ" in log['message']]
        
        print("\n📊 ОТЧЁТ АУДИТА")
        print(f"Всего логов: {len(all_logs)}")
        print(f"Предупреждений: {len(warnings)}")
        print(f"Заблокировано: {len(suspicious)}")

class Client:
    def __init__(self, id: str, name: str):
        self.id = id
        self.name = name
        self.accounts = []
        
class SimpleBank:
    def __init__(self):
        self.clients = {}
        self.accounts = {}
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
        client = self.clients.get(client_id)
        if not client:
            print("❌ Клиент не найден")
            return
        print(f"\n👤 {client.name} (ID: {client.id})")
        print(f"Счета: {', '.join(client.accounts)}")
    
    def client_history(self, client_id: str):
        client_tx = [tx for tx in self.transactions if tx.from_client == self.clients[client_id].name]
        blocked = [tx for tx in client_tx if tx.blocked]
        print(f"\n📋 {self.clients[client_id].name}: {len(client_tx)} tx")
        print(f"Заблокировано: {len(blocked)}")
        if blocked:
            print("🚨 Заблокированные:")
            for tx in blocked:
                print(f"  {tx.id}: {tx.amount}")

    def final_report(self):
        print("\n🏆 ФИНАЛЬНЫЙ ОТЧЁТ БАНКА")
        total_tx = len(self.transactions)
        blocked_tx = len([tx for tx in self.transactions if tx.blocked])
        print(f"📈 Всего tx: {total_tx}")
        print(f"🚫 Заблокировано: {blocked_tx} ({blocked_tx/total_tx*100:.1f}%)")
        
        client_tx_count = {}
        for tx in self.transactions:
            client_tx_count[tx.from_client] = client_tx_count.get(tx.from_client, 0) + 1
        
        top_clients = sorted(client_tx_count.items(), key=lambda x: x[1], reverse=True)[:3]
        print("🥇 Топ-3 активных клиентов:")
        for i, (name, count) in enumerate(top_clients, 1):
            print(f"  {i}. {name}: {count} tx")
        
        total_balance = sum(self.accounts.values())
        print(f"💰 Общий баланс банка: {total_balance:,.0f}")
        
        reporter = AuditReporter(self.risk_analyzer)
        reporter.report()

def simulate_transactions(bank: SimpleBank, count: int = 50):
    import random
    
    clients = list(bank.clients.keys())
    accounts = list(bank.accounts.keys())
    
    for i in range(count):
        client_from = random.choice(clients)
        account_to = random.choice(accounts)
        amount = random.uniform(100, 50000)
        
        is_suspicious = random.random() < 0.2
        if is_suspicious:
            amount = random.uniform(100000, 500000)
        
        is_night = random.random() < 0.1
        is_new = account_to not in ["acc1", "acc2"]
        
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

# ReportBuilder БЕЗ pandas
class ReportBuilder:
    def __init__(self, bank: SimpleBank):
        self.bank = bank
        os.makedirs('charts', exist_ok=True)
        os.makedirs('reports', exist_ok=True)

    def _get_client_data(self, client_id: str):
        client = self.bank.clients.get(client_id)
        if not client:
            return {}
        client_txs = [tx for tx in self.bank.transactions if tx.from_client == client.name]
        blocked_txs = [tx for tx in client_txs if tx.blocked]
        balance = sum(self.bank.accounts.get(acc, 0) for acc in client.accounts)
        return {
            'client_id': client_id,
            'name': client.name,
            'accounts_count': len(client.accounts),
            'total_balance': balance,
            'tx_count': len(client_txs),
            'blocked_txs': len(blocked_txs),
            'avg_tx_amount': sum(tx.amount for tx in client_txs) / len(client_txs) if client_txs else 0
        }

    def generate_report(self, report_type='bank', client_id=None):
        if report_type == 'client' and client_id:
            data = [self._get_client_data(client_id)]
            title = f"ОТЧЁТ ПО КЛИЕНТУ {client_id}"
        elif report_type == 'bank':
            total_balance = sum(self.bank.accounts.values())
            total_blocked = len([tx for tx in self.bank.transactions if tx.blocked])
            total_tx = len(self.bank.transactions)
            data = [{
                'total_clients': len(self.bank.clients),
                'total_accounts': len(self.bank.accounts),
                'total_balance': total_balance,
                'total_transactions': total_tx,
                'blocked_transactions': total_blocked,
                'block_rate_percent': round((total_blocked / total_tx * 100), 2) if total_tx else 0
            }]
            title = "ОТЧЁТ ПО БАНКУ"
        elif report_type == 'risk':
            warnings = self.bank.risk_analyzer.audit.filter(AuditLevel.WARNING)
            suspicious = len([log for log in warnings if "БЛОКИРУЕМ" in log['message']])
            data = [{
                'total_warnings': len(warnings),
                'blocked_transactions': suspicious,
                'warning_types': dict(Counter([log['message'][:30] for log in warnings[:10]]))
            }]
            title = "ОТЧЁТ ПО РИСКАМ"
        else:
            return "❌ Неизвестный тип отчёта"

        print(f"\n{title}")
        for item in data:
            for key, value in item.items():
                print(f"  {key}: {value}")
        
        return data

    def export_to_json(self, data, filename):
        filepath = f"reports/{filename}.json"
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)
        print(f"✅ JSON: {filepath}")

    def export_to_csv(self, data, filename):
        filepath = f"reports/{filename}.csv"
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            if isinstance(data, list) and data:
                writer = csv.DictWriter(f, fieldnames=data[0].keys())
                writer.writeheader()
                writer.writerows(data)
            else:
                writer = csv.DictWriter(f, fieldnames=data.keys())
                writer.writeheader()
                writer.writerow(data)
        print(f"✅ CSV: {filepath}")

    def create_charts(self):
        print("\n📈 Генерация графиков...")

        # 1. Круговая: доля баланса топ-5 клиентов
        client_balances = {}
        for client_id, client in self.bank.clients.items():
            balance = sum(self.bank.accounts.get(acc, 0) for acc in client.accounts)
            client_balances[client.name[:12]] = balance
        
        top5 = dict(sorted(client_balances.items(), key=lambda x: x[1], reverse=True)[:5])
        if top5:
            fig1, ax1 = plt.subplots(figsize=(8, 6))
            sizes = list(top5.values())
            labels = list(top5.keys())
            wedges, texts, autotexts = ax1.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
            ax1.set_title('Доля баланса по топ-5 клиентам')
            self._save_chart(fig1, 'pie_client_balances')

        # 2. Столбцы: статусы tx
        total_tx = len(self.bank.transactions)
        blocked_tx = len([tx for tx in self.bank.transactions if tx.blocked])
        fig2, ax2 = plt.subplots(figsize=(7, 5))
        categories = ['Выполнено', 'Заблокировано']
        values = [total_tx - blocked_tx, blocked_tx]
        bars = ax2.bar(categories, values, color=['#4CAF50', '#F44336'])
        ax2.set_title('Транзакции по статусам')
        ax2.set_ylabel('Количество')
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + total_tx*0.01,
                    f'{int(height)}', ha='center', va='bottom', fontweight='bold')
        self._save_chart(fig2, 'bar_tx_status')

        # 3. Линия: симуляция движения баланса
        n_points = min(30, len(self.bank.transactions))
        times = [datetime.now() - timedelta(minutes=i*30) for i in range(n_points)][::-1]
        base_balance = sum(self.bank.accounts.values())
        changes = np.cumsum(np.random.normal(500, 2000, n_points))
        balances = base_balance + changes
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        ax3.plot(times, balances, marker='o', linewidth=2, markersize=4, color='#2196F3')
        ax3.set_title('Динамика общего баланса банка')
        ax3.set_xlabel('Время')
        ax3.set_ylabel('Баланс')
        ax3.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        self._save_chart(fig3, 'line_balance_movement')

        print("✅ Все 3 графика сохранены в charts/")

    def _save_chart(self, fig, filename):
        fig.savefig(f"charts/{filename}.png", dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)

# === ДЕМО + ТЕСТИРОВАНИЕ ===
if __name__ == "__main__":
    print("🏦 === Lesson7.py: Полная система БЕЗ pandas ===\n")
    
    bank = SimpleBank()
    
    clients_data = [
        ("1", "Иванов Иван"), ("2", "Мария Петрова"), ("3", "Петров Сергей"),
        ("4", "Сидорова Анна"), ("5", "Козлов Дмитрий"), ("6", "Смирнова Ольга"),
        ("7", "Васильев Алексей"), ("8", "Новикова Екатерина"), ("9", "Морозов Олег"),
        ("10", "Федорова Елена")
    ]
    
    for client_id, name in clients_data:
        client = Client(client_id, name)
        bank.add_client(client)
    
    account_data = [
        ("1", "acc001", 100000), ("1", "acc002", 50000),
        ("2", "acc003", 75000), 
        ("3", "acc004", 20000), ("3", "acc005", 30000),
        ("4", "acc006", 150000), ("5", "acc007", 80000),
        ("6", "acc008", 120000), ("6", "acc009", 40000),
        ("7", "acc010", 95000), ("8", "acc011", 60000), ("9", "acc012", 110000)
    ]
    
    for client_id, acc_num, balance in account_data:
        bank.open_account(client_id, acc_num, balance)
    
    simulate_transactions(bank, 50)
    
    bank.show_client("1")
    bank.client_history("1")
    
    print("\n" + "="*70)
    print("📋 ReportBuilder: ОТЧЁТЫ + ГРАФИКИ")
    print("="*70)
    
    builder = ReportBuilder(bank)
    
    # Генерация всех отчётов
    print("\n🏛️  БАНК:")
    bank_data = builder.generate_report('bank')
    builder.export_to_json(bank_data, 'report_bank')
    builder.export_to_csv(bank_data, 'report_bank')
    
    print("\n👤 КЛИЕНТ 1:")
    client_data = builder.generate_report('client', '1')
    builder.export_to_json(client_data, 'report_client1')
    builder.export_to_csv(client_data, 'report_client1')
    
    print("\n⚠️  РИСКИ:")
    risk_data = builder.generate_report('risk')
    builder.export_to_json(risk_data, 'report_risk')
    builder.export_to_csv(risk_data, 'report_risk')
    
    # Графики
    builder.create_charts()
    
    bank.final_report()
    
    print("\n🎉 ✅ День 7 ГОТОВ! Проверь:")
    print("   📁 reports/ - JSON+CSV")
    print("   📁 charts/  - 3 PNG графика")
    print("   📄 audit.log")
