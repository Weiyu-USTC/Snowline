import Foundation

// Run against the same policy compiled into the widget. No network calls,
// changes to the Mac's clock, or saved preferences are required.
@main
enum HourlyRefreshPolicyTests {
    static func date(_ value: String) -> Date {
        let format = ISO8601DateFormatter()
        return format.date(from: value)!
    }

    static func calendar(_ zone: String) -> Calendar {
        var value = Calendar(identifier: .gregorian)
        value.timeZone = TimeZone(identifier: zone)!
        return value
    }

    static func main() {
        var checks = 0
        func expect<T: Equatable>(_ actual: T, _ expected: T, _ name: String) {
            guard actual == expected else {
                fatalError("FAILED: \(name) — got \(actual), expected \(expected)")
            }
            checks += 1
        }
        let ny = calendar("America/New_York")
        let utc = calendar("UTC")
        let slot = date("2026-11-13T17:00:00Z")

        expect(HourlyRefreshPolicy.nextBoundary(after: date("2026-11-13T16:14:35Z"), calendar: ny),
               slot, "Next local top of hour")
        expect(HourlyRefreshPolicy.nextBoundary(after: slot, calendar: ny),
               date("2026-11-13T18:00:00Z"), "No immediate catch-up when enabled exactly at a boundary")
        expect(HourlyRefreshPolicy.nextBoundary(after: slot.addingTimeInterval(-0.001), calendar: ny),
               slot, "Fractional second before the hour")
        expect(HourlyRefreshPolicy.nextBoundary(after: slot.addingTimeInterval(0.001), calendar: ny),
               date("2026-11-13T18:00:00Z"), "Fractional second after the hour")
        expect(HourlyRefreshPolicy.nextBoundary(after: date("2026-11-14T04:50:00Z"), calendar: ny),
               date("2026-11-14T05:00:00Z"), "Local midnight")
        expect(HourlyRefreshPolicy.nextBoundary(after: date("2026-11-13T12:40:00Z"), calendar: calendar("Asia/Kolkata")),
               date("2026-11-13T13:30:00Z"), "Half-hour timezone uses computer wall clock")
        expect(HourlyRefreshPolicy.nextBoundary(after: date("2026-11-13T12:30:00Z"), calendar: calendar("Asia/Kathmandu")),
               date("2026-11-13T13:15:00Z"), "Quarter-hour timezone uses computer wall clock")
        expect(HourlyRefreshPolicy.nextBoundary(after: date("2026-11-01T04:30:00Z"), calendar: ny),
               date("2026-11-01T05:00:00Z"), "First autumn 1 AM")
        expect(HourlyRefreshPolicy.nextBoundary(after: date("2026-11-01T05:30:00Z"), calendar: ny),
               date("2026-11-01T06:00:00Z"), "Repeated autumn 1 AM is a distinct hour")
        expect(HourlyRefreshPolicy.nextBoundary(after: date("2026-11-01T06:30:00Z"), calendar: ny),
               date("2026-11-01T07:00:00Z"), "Hour after autumn clock change")
        expect(HourlyRefreshPolicy.nextBoundary(after: date("2026-03-08T06:30:00Z"), calendar: ny),
               date("2026-03-08T07:00:00Z"), "Spring jump skips nonexistent 2 AM")
        expect(HourlyRefreshPolicy.nextBoundary(after: date("2026-11-13T16:14:35Z"), calendar: utc),
               slot, "Explicit new timezone reschedules correctly")

        func decision(_ now: Date = slot, scheduled: Date = slot,
                      enabled: Bool = true, wifi: Bool = true, busy: Bool = false,
                      last: Date? = nil, cal: Calendar = ny) -> HourlyRefreshDecision {
            HourlyRefreshPolicy.decision(now: now, scheduled: scheduled,
                                         enabled: enabled, wifiConnected: wifi,
                                         busy: busy, lastAttempt: last, calendar: cal)
        }
        expect(decision(), .run, "Connected at HH:00:00 starts the refresh")
        expect(decision(slot.addingTimeInterval(0.999)), .run, "Timer dispatch within the boundary second")
        expect(decision(slot.addingTimeInterval(-0.001)), .outsideBoundary, "Early callback cannot refresh")
        expect(decision(slot.addingTimeInterval(1)), .outsideBoundary, "Late callback cannot refresh")
        expect(decision(slot.addingTimeInterval(3600)), .outsideBoundary, "Wake does not replay a missed hour")
        expect(decision(enabled: false), .disabled, "Disabled option never refreshes")
        expect(decision(wifi: false), .noWiFi, "Missing or unknown Wi-Fi skips this hour")
        expect(decision(busy: true), .busy, "Existing update prevents a second request")
        expect(decision(last: slot), .duplicate, "Persisted slot prevents duplicate after restart")
        expect(decision(last: slot.addingTimeInterval(-3600)), .run, "Previous hour does not suppress next hour")
        let repeated = date("2026-11-01T06:00:00Z")
        expect(decision(repeated, scheduled: repeated, last: date("2026-11-01T05:00:00Z")),
               .run, "Repeated wall-clock label does not count as duplicate")
        let offMinute = slot.addingTimeInterval(60)
        expect(decision(offMinute, scheduled: offMinute), .outsideBoundary, "Only minute zero is eligible")
        let offSecond = slot.addingTimeInterval(1)
        expect(decision(offSecond, scheduled: offSecond), .outsideBoundary, "Only second zero is eligible")
        let fractionalSlot = slot.addingTimeInterval(0.5)
        expect(decision(fractionalSlot, scheduled: fractionalSlot), .outsideBoundary, "Scheduled boundary must be exact")
        expect(decision(slot.addingTimeInterval(300), wifi: true), .outsideBoundary,
               "Wi-Fi returning later does not cause an off-hour catch-up")
        let kathmanduSlot = date("2026-11-13T13:15:00Z")
        expect(decision(kathmanduSlot, scheduled: kathmanduSlot, cal: calendar("Asia/Kathmandu")),
               .run, "Local top of hour can differ from UTC top of hour")

        print("Passed \(checks) hourly Auto-update policy checks.")
    }
}
