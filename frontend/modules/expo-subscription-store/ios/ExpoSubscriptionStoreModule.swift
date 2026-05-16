import ExpoModulesCore
import SwiftUI
import StoreKit

public class ExpoSubscriptionStoreModule: Module {
  public func definition() -> ModuleDefinition {
    Name("ExpoSubscriptionStore")

    AsyncFunction("presentSubscriptionStore") { (groupID: String) in
      await MainActor.run {
        guard let scene = UIApplication.shared.connectedScenes.first as? UIWindowScene,
              let root = scene.windows.first(where: { $0.isKeyWindow })?.rootViewController else {
          return
        }

        // Find the topmost presented controller
        var topController = root
        while let presented = topController.presentedViewController {
          topController = presented
        }

        if #available(iOS 17.0, *) {
          let hosting = UIHostingController(
            rootView: SubscriptionStoreView(groupID: groupID)
              .subscriptionStoreControlStyle(.automatic)
          )
          hosting.modalPresentationStyle = .pageSheet
          topController.present(hosting, animated: true)
        } else {
          // Fallback for iOS 16 and below - just return, let JS handle it
          return
        }
      }
    }

    Function("isSubscriptionStoreAvailable") { () -> Bool in
      if #available(iOS 17.0, *) {
        return true
      }
      return false
    }
  }
}
