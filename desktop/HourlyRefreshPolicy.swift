import Foundation

enum HourlyRefreshDecision: String {
    case run, disabled, noWiFi, busy, duplicate, outsideBoundary
}

struct HourlyRefreshPolicy {
    static func nextBoundary(after date: Date, calendar: Calendar = .autoupdatingCurrent) -> Date {
        // Match the computer's local clock, including zones with non-hour offsets.
        calendar.nextDate(after: date, matching: DateComponents(minute: 0, second: 0),
                          matchingPolicy: .nextTime, repeatedTimePolicy: .first)!
    }

    static func decision(now: Date, scheduled: Date, enabled: Bool, wifiConnected: Bool,
                         busy: Bool, lastAttempt: Date?, calendar: Calendar = .autoupdatingCurrent) -> HourlyRefreshDecision {
        guard enabled else { return .disabled }
        let clock = calendar.dateComponents([.minute, .second, .nanosecond], from: scheduled)
        let delay = now.timeIntervalSince(scheduled)
        guard clock.minute == 0, clock.second == 0, clock.nanosecond == 0,
              delay >= 0, delay < 1 else { return .outsideBoundary }
        guard lastAttempt != scheduled else { return .duplicate }
        guard wifiConnected else { return .noWiFi }
        guard !busy else { return .busy }
        return .run
    }
}
