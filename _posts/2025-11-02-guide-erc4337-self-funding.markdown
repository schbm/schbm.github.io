---
layout: single
title:  "A Guide to Self-Funding Smart-Contracts"
date:   2025-11-02 14:00:00 +0100
show_date: true
categories: blockchain guide
tags: blockchain ethereum erc-4337 account-abstraction
toc: true
---

To improve UX many contract devs may want to use account-abstraction to enable functionalities like:
- Social Logins
- Account Recovery
- Gas Sponsorship
- and more...

Within Ethereum the ERC-4337 standard defines the interface and the infrastructure needed to accomplish this.
This guide will show how to quickly spin up a contract prototype. Tho I will not go into the details on the topics like authentication and authorization
which would be essential in a real life deployment scenario!

# ERC-4337
There is already extensive information available about this specification, so I will only provide a brief summary.
ERC-4337 defines that user operations are processed off-chain by a bundler and, once validated, are submitted on-chain through an EntryPoint.
The components involved are illustrated in the following figure:
![ERC-4337](https://cdn.prod.website-files.com/66ec556d91c3ab378f61fadf/66ec55add0430f1f5803e751_641bcdb974e0977985f925dc_63f13a9ae1e45ac6b83ed3a6_components-erc-4337.svg)

The proposal defines new signature interfaces for smart contracts:
```
function validateUserOp(
        PackedUserOperation calldata userOp,
        bytes32 userOpHash,
        uint256 missingAccountFunds
    ) external override returns (uint256 validationData) {}

function execute(address dest, uint256 value, bytes calldata func) external {}
```

As in other proposals, any contract that implements these functions is compatible with the standard.
Within the infrastructure, the process works roughly as follows:
1. A user processes its task as an `UserOperation`, that includes about 14 additional fields specific to AA. The `UserOperation` is not yet submitted to Ethereum.
2. The user signs it and passes it to a bundler. There are many bundlers like for example Alchemy or Pimlico. These typically provide RPC interfaces, which are often compatible with Viem.
So not much modification is needed.
3. The bundler performs validation checks. (Optionally) Creates optimized batches (bundles 😜) to minimize gas and prepares them for the EntryPoint.
4. The [EntryPoint](https://etherscan.io/address/0x4337084D9E255Ff0702461CF8895CE9E3b5Ff108#code) then executes the Operations on the chain. The `UserOperation` is processed in 2 atomic steps: Validation via `validateUserOp()` and execution via `execute()`. There exist official EntryPoint contracts which can be found online.

# Our Example Project
For our example, we assume an existing smart contract that provides a voting service.
Because there are already many tutorials on basic voting contracts, we'll add extra constraints:
- We will introduce ERC-4337 account abstraction to enable gas sponsorship now and support additional features later.
- The design must use a single contract (no multi-contract architecture).
- We will not rely on an external Paymaster.

## Implementation
We start by importing the battle-tested OpenZeppelin definitions
```solidity 
import "@account-abstraction/contracts/interfaces/IAccount.sol";
import "@account-abstraction/contracts/interfaces/IEntryPoint.sol";

// for interpreting the signer address
import "@openzeppelin/contracts/utils/cryptography/ECDSA.sol";
import "@openzeppelin/contracts/utils/cryptography/MessageHashUtils.sol";

// In this example we may want to stay in control of the contract
// We can do this by setting ourselves as the owner once we deploy the contract
import "@openzeppelin/contracts/access/Ownable.sol";
```

And implement the interface for our contract:
```solidity 
contract SophisticatedVoting is IAccount, Ownable {
    // For the helper functions
    using ECDSA for bytes32;
    using MessageHashUtils for bytes32;


    // We define the official onchain EntryPoint
    IEntryPoint public immutable entryPoint = IEntryPoint(0x0000000071727De22E5E9d8BAf0edAc6f37da032);

    // We add nonces to guard against different vulnerabilities like replay-attacks:
    uint256 public ownerNonce; // A seperate owner nonce is not a must but helps during debugging
    mapping(address => uint256) public nonces;

    // We set ourselves as the owner
    constructor() Ownable(msg.sender) {}

    // We add a way to receive payments
    // These funds will then be used to pay the gas fees
    receive() external payable {}

    // Add the ERC-4337 functions we need to specify
    function validateUserOp(
        PackedUserOperation calldata userOp,
        bytes32 userOpHash,
        uint256 missingAccountFunds
    ) external override returns (uint256 validationData) {}
    function execute(address dest, uint256 value, bytes calldata func) external {}
}
```

We now implement the validation function:
```solidity 
function validateUserOp(
    PackedUserOperation calldata userOp,
    bytes32 userOpHash,
    uint256 missingAccountFunds
) external override returns (uint256 validationData) {

    // only the official EntryPoint may run this function
    require(msg.sender == address(entryPoint), "Only EntryPoint");

    // we can get the original signer address like this:
    bytes32 hash = userOpHash.toEthSignedMessageHash();
    address signer = hash.recover(userOp.signature); // this is the address of the EOA that signed the action

    // Here comes the hard part
    // You need to implement a custom authentication and authorization check here
    // Tho this will depend on your specific use-case
    // One can also use OpenZeppelin definitions to make like easier like for example role based permissions!

    // Increment the nonce!
    if (isOwner) {
        if (userOp.nonce != ownerNonce) return SIG_VALIDATION_FAILED;
        unchecked { ownerNonce++; }
    } else {
        uint256 expected = nonces[signer];
        if (userOp.nonce != expected) return SIG_VALIDATION_FAILED;
        unchecked { nonces[signer] = expected + 1; }
    }

    // no pay back the owed fees!
    if (missingAccountFunds > 0) {
        // Message sender here will be the EntryPoint
        (bool success,) = payable(msg.sender).call{value: missingAccountFunds}("");
        require(success, "Failed to pay");
    }
    return 0;
}
```

Now for the execute function we keep it simple:
```solidity
function execute(address dest, uint256 value, bytes calldata func) external {
    // Ensure only the EntryPoint can execute this
    require(msg.sender == address(entryPoint), "Only EntryPoint");
    // Execute the function
    (bool success,) = dest.call{value: value}(func);
    require(success, "Execution failed");
}
```

With this setupt one problem arise.
Suppose we have a function in our service where the original sender of the action is needed.
How would we get his address?

The simplest solution is the following:
```solidity
// add a temporary variable that hold the original signer
address private _aaSigner;
// and a helper function that either returns him if available
// or otherwise returns the sender address
function _currentActor() internal view returns (address) {
    return _aaSigner != address(0) ? _aaSigner : msg.sender;
}
```

Now within our validation and execution logic add the following at the end:
```solidity
function validateUserOp(
    PackedUserOperation calldata userOp,
    bytes32 userOpHash,
    uint256 missingAccountFunds
) external override returns (uint256 validationData) {
    
    // validation logic...

    _aaSigner = signer;

    return 0;
    }

function execute(address dest, uint256 value, bytes calldata func) external {
    
    // execution logic

    _aaSigner = address(0);

    require(success, "Execution failed");
}
```

Within our services we can now fetch the original address for example like this:
```solidity
function vote(uint256 pollId, uint256 optionIndex) public {
        address voter = _currentActor();
        // our voting logic...
    }
```

**Attention!** This guide skips critical access control and security measures.
Deploying the contract as-is can expose you to severe security risks.
{: .notice--danger}

And that's it, the contract side is complete. The remaining work involves building the user-facing application.
Fortunately, much of the implementation can be simplified by leveraging the utilities provided by Viem.
For example by using the [BundlerClient](https://viem.sh/account-abstraction/clients/bundler) as a generic interface for Bundlers:
```javascript
const client = createPublicClient({
  chain: mainnet,
  transport: http()
})
 
const bundlerClient = createBundlerClient({ 
  client, 
  transport: http('https://public.pimlico.io/v2/1/rpc') 
}) 

// then execute
const hash = await bundlerClient.sendUserOperation({ 
  account,
  calls: [{
    to: '0x70997970c51812dc3a010c7d01b50e0d17dc79c8',
    value: parseEther('1')
  }],
})
```