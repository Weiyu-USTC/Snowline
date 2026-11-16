import AppKit
import Network
import SwiftUI

@MainActor
final class HourlyAutoUpdater: ObservableObject {
    @Published var enabled: Bool {
        didSet {
            UserDefaults.standard.set(enabled, forKey: "hourlyAutoUpdate")
            updateActivity()
            reschedule()
            record(enabled ? "enabled" : "disabled")
        }
    }
    @Published private(set) var wifiConnected = false
    @Published private(set) var nextUpdate: Date?
    private let monitor = NWPathMonitor(requiredInterfaceType: .wifi)
    private var timer: Timer?
    private var clockObservers: [NSObjectProtocol] = []
    private var wakeObserver: NSObjectProtocol?
    private var sleepObserver: NSObjectProtocol?
    private var activity: NSObjectProtocol?
    private var sleeping = false
    private var lastAttempt: Date?
    private var busy: () -> Bool = { true }
    private var refresh: () async -> Void = {}
    private var logURL: URL?

    init() {
        enabled = UserDefaults.standard.bool(forKey: "hourlyAutoUpdate")
        if let stamp = UserDefaults.standard.object(forKey: "hourlyAutoUpdateLastAttempt") as? Double {
            lastAttempt = Date(timeIntervalSince1970: stamp)
        }
    }

    func start(logURL: URL, isBusy: @escaping () -> Bool, refresh: @escaping () async -> Void) {
        self.logURL = logURL
        self.busy = isBusy
        self.refresh = refresh
        monitor.pathUpdateHandler = { [weak self] path in
            let connected = path.status == .satisfied && path.usesInterfaceType(.wifi)
            Task { @MainActor in
                self?.wifiConnected = connected
                // A reconnect never causes a missed-hour download.
            }
        }
        monitor.start(queue: DispatchQueue(label: "snowline.wifi"))
        for name in [Notification.Name.NSSystemClockDidChange, .NSSystemTimeZoneDidChange] {
            clockObservers.append(NotificationCenter.default.addObserver(forName: name, object: nil, queue: .main) { [weak self] _ in
                Task { @MainActor in self?.reschedule() }
            })
        }
        sleepObserver = NSWorkspace.shared.notificationCenter.addObserver(forName: NSWorkspace.willSleepNotification, object: nil, queue: .main) { [weak self] _ in
            Task { @MainActor in
                self?.sleeping = true
                self?.timer?.invalidate()
                self?.timer = nil
                self?.nextUpdate = nil
            }
        }
        wakeObserver = NSWorkspace.shared.notificationCenter.addObserver(forName: NSWorkspace.didWakeNotification, object: nil, queue: .main) { [weak self] _ in
            Task { @MainActor in
                self?.sleeping = false
                self?.reschedule()
            }
        }
        updateActivity()
        reschedule()
    }

    var statusText: String {
        guard enabled else { return "" }
        guard let nextUpdate = nextUpdate else { return "Resumes next hour after wake" }
        let formatter = DateFormatter()
        formatter.timeZone = .autoupdatingCurrent
        formatter.dateFormat = "h:mm a"
        let next = formatter.string(from: nextUpdate)
        return wifiConnected ? "Next \(next) · Wi-Fi" : "No Wi-Fi · checks \(next)"
    }

    // Rechecked immediately before the automatic POST, after service startup.
    var mayStartRefresh: Bool {
        enabled && !sleeping && monitor.currentPath.status == .satisfied && monitor.currentPath.usesInterfaceType(.wifi)
    }

    private func updateActivity() {
        if enabled && activity == nil {
            // Keep the hourly timer responsive when collapsed without preventing Mac sleep.
            activity = ProcessInfo.processInfo.beginActivity(options: .userInitiatedAllowingIdleSystemSleep,
                                                              reason: "Snowline hourly Wi-Fi updates")
        } else if !enabled, let activity = activity {
            ProcessInfo.processInfo.endActivity(activity)
            self.activity = nil
        }
    }

    private func reschedule() {
        timer?.invalidate()
        timer = nil
        nextUpdate = nil
        guard enabled && !sleeping else { return }
        let boundary = HourlyRefreshPolicy.nextBoundary(after: Date())
        nextUpdate = boundary
        let nextTimer = Timer(fire: boundary, interval: 0, repeats: false) { [weak self] _ in
            Task { @MainActor in self?.fire(boundary) }
        }
        nextTimer.tolerance = 0
        timer = nextTimer
        RunLoop.main.add(nextTimer, forMode: .common)
    }

    private func fire(_ boundary: Date) {
        let now = Date()
        let decision = HourlyRefreshPolicy.decision(now: now, scheduled: boundary,
            enabled: enabled && !sleeping, wifiConnected: mayStartRefresh,
            busy: busy(), lastAttempt: lastAttempt)
        reschedule()
        record(decision.rawValue, scheduled: boundary, at: now)
        guard decision == .run else { return }
        // Persist before starting asynchronous work to prevent duplicate boundary attempts.
        lastAttempt = boundary
        UserDefaults.standard.set(boundary.timeIntervalSince1970, forKey: "hourlyAutoUpdateLastAttempt")
        Task { await refresh() }
    }

    private func record(_ outcome: String, scheduled: Date? = nil, at now: Date = Date()) {
        guard let logURL = logURL else { return }
        let formatter = ISO8601DateFormatter()
        var entry: [String: Any] = ["event": outcome, "at": formatter.string(from: now),
            "enabled": enabled, "wifi_connected": wifiConnected,
            "computer_timezone": TimeZone.autoupdatingCurrent.identifier]
        if let scheduled = scheduled { entry["scheduled_for"] = formatter.string(from: scheduled) }
        if let nextUpdate = nextUpdate { entry["next_update"] = formatter.string(from: nextUpdate) }
        if let data = try? JSONSerialization.data(withJSONObject: entry, options: [.prettyPrinted, .sortedKeys]) {
            try? FileManager.default.createDirectory(at: logURL.deletingLastPathComponent(), withIntermediateDirectories: true)
            try? data.write(to: logURL, options: .atomic)
        }
    }
}
